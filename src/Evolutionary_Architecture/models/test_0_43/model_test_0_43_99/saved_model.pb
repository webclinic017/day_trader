Аю'
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8█С&
z
dense_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`* 
shared_namedense_10/kernel
s
#dense_10/kernel/Read/ReadVariableOpReadVariableOpdense_10/kernel*
_output_shapes

:`*
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
У
lstm_20/lstm_cell_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]°*,
shared_namelstm_20/lstm_cell_20/kernel
М
/lstm_20/lstm_cell_20/kernel/Read/ReadVariableOpReadVariableOplstm_20/lstm_cell_20/kernel*
_output_shapes
:	]°*
dtype0
и
%lstm_20/lstm_cell_20/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ю°*6
shared_name'%lstm_20/lstm_cell_20/recurrent_kernel
б
9lstm_20/lstm_cell_20/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_20/lstm_cell_20/recurrent_kernel* 
_output_shapes
:
Ю°*
dtype0
Л
lstm_20/lstm_cell_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:°**
shared_namelstm_20/lstm_cell_20/bias
Д
-lstm_20/lstm_cell_20/bias/Read/ReadVariableOpReadVariableOplstm_20/lstm_cell_20/bias*
_output_shapes	
:°*
dtype0
Ф
lstm_21/lstm_cell_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЮА*,
shared_namelstm_21/lstm_cell_21/kernel
Н
/lstm_21/lstm_cell_21/kernel/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell_21/kernel* 
_output_shapes
:
ЮА*
dtype0
з
%lstm_21/lstm_cell_21/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`А*6
shared_name'%lstm_21/lstm_cell_21/recurrent_kernel
а
9lstm_21/lstm_cell_21/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_21/lstm_cell_21/recurrent_kernel*
_output_shapes
:	`А*
dtype0
Л
lstm_21/lstm_cell_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А**
shared_namelstm_21/lstm_cell_21/bias
Д
-lstm_21/lstm_cell_21/bias/Read/ReadVariableOpReadVariableOplstm_21/lstm_cell_21/bias*
_output_shapes	
:А*
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
И
Adam/dense_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/dense_10/kernel/m
Б
*Adam/dense_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/m*
_output_shapes

:`*
dtype0
А
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
б
"Adam/lstm_20/lstm_cell_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]°*3
shared_name$"Adam/lstm_20/lstm_cell_20/kernel/m
Ъ
6Adam/lstm_20/lstm_cell_20/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_20/lstm_cell_20/kernel/m*
_output_shapes
:	]°*
dtype0
╢
,Adam/lstm_20/lstm_cell_20/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ю°*=
shared_name.,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m
п
@Adam/lstm_20/lstm_cell_20/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m* 
_output_shapes
:
Ю°*
dtype0
Щ
 Adam/lstm_20/lstm_cell_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*1
shared_name" Adam/lstm_20/lstm_cell_20/bias/m
Т
4Adam/lstm_20/lstm_cell_20/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_20/lstm_cell_20/bias/m*
_output_shapes	
:°*
dtype0
в
"Adam/lstm_21/lstm_cell_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЮА*3
shared_name$"Adam/lstm_21/lstm_cell_21/kernel/m
Ы
6Adam/lstm_21/lstm_cell_21/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_21/lstm_cell_21/kernel/m* 
_output_shapes
:
ЮА*
dtype0
╡
,Adam/lstm_21/lstm_cell_21/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`А*=
shared_name.,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m
о
@Adam/lstm_21/lstm_cell_21/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m*
_output_shapes
:	`А*
dtype0
Щ
 Adam/lstm_21/lstm_cell_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_21/lstm_cell_21/bias/m
Т
4Adam/lstm_21/lstm_cell_21/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_21/lstm_cell_21/bias/m*
_output_shapes	
:А*
dtype0
И
Adam/dense_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:`*'
shared_nameAdam/dense_10/kernel/v
Б
*Adam/dense_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_10/kernel/v*
_output_shapes

:`*
dtype0
А
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
б
"Adam/lstm_20/lstm_cell_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]°*3
shared_name$"Adam/lstm_20/lstm_cell_20/kernel/v
Ъ
6Adam/lstm_20/lstm_cell_20/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_20/lstm_cell_20/kernel/v*
_output_shapes
:	]°*
dtype0
╢
,Adam/lstm_20/lstm_cell_20/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ю°*=
shared_name.,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v
п
@Adam/lstm_20/lstm_cell_20/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v* 
_output_shapes
:
Ю°*
dtype0
Щ
 Adam/lstm_20/lstm_cell_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:°*1
shared_name" Adam/lstm_20/lstm_cell_20/bias/v
Т
4Adam/lstm_20/lstm_cell_20/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_20/lstm_cell_20/bias/v*
_output_shapes	
:°*
dtype0
в
"Adam/lstm_21/lstm_cell_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ЮА*3
shared_name$"Adam/lstm_21/lstm_cell_21/kernel/v
Ы
6Adam/lstm_21/lstm_cell_21/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_21/lstm_cell_21/kernel/v* 
_output_shapes
:
ЮА*
dtype0
╡
,Adam/lstm_21/lstm_cell_21/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	`А*=
shared_name.,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v
о
@Adam/lstm_21/lstm_cell_21/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v*
_output_shapes
:	`А*
dtype0
Щ
 Adam/lstm_21/lstm_cell_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*1
shared_name" Adam/lstm_21/lstm_cell_21/bias/v
Т
4Adam/lstm_21/lstm_cell_21/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_21/lstm_cell_21/bias/v*
_output_shapes	
:А*
dtype0

NoOpNoOp
║6
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ї5
valueы5Bш5 Bс5
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
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
l
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
╨
&iter

'beta_1

(beta_2
	)decay
*learning_rate mp!mq+mr,ms-mt.mu/mv0mw vx!vy+vz,v{-v|.v}/v~0v
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
8
+0
,1
-2
.3
/4
05
 6
!7
н
regularization_losses
	variables
1metrics
2layer_metrics
3layer_regularization_losses

4layers
5non_trainable_variables
	trainable_variables
 
О
6
state_size

+kernel
,recurrent_kernel
-bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
 
 

+0
,1
-2

+0
,1
-2
╣
regularization_losses
	variables
;metrics

<states
=layer_metrics
>layer_regularization_losses

?layers
@non_trainable_variables
trainable_variables
 
 
 
н
regularization_losses
Ametrics
	variables
Blayer_metrics
Clayer_regularization_losses
trainable_variables
Dnon_trainable_variables

Elayers
О
F
state_size

.kernel
/recurrent_kernel
0bias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
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
╣
regularization_losses
	variables
Kmetrics

Lstates
Mlayer_metrics
Nlayer_regularization_losses

Olayers
Pnon_trainable_variables
trainable_variables
 
 
 
н
regularization_losses
Qmetrics
	variables
Rlayer_metrics
Slayer_regularization_losses
trainable_variables
Tnon_trainable_variables

Ulayers
[Y
VARIABLE_VALUEdense_10/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_10/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
н
"regularization_losses
Vmetrics
#	variables
Wlayer_metrics
Xlayer_regularization_losses
$trainable_variables
Ynon_trainable_variables

Zlayers
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
WU
VARIABLE_VALUElstm_20/lstm_cell_20/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_20/lstm_cell_20/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_20/lstm_cell_20/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUElstm_21/lstm_cell_21/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%lstm_21/lstm_cell_21/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElstm_21/lstm_cell_21/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
 

+0
,1
-2

+0
,1
-2
н
7regularization_losses
]metrics
8	variables
^layer_metrics
_layer_regularization_losses
9trainable_variables
`non_trainable_variables

alayers
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
 

.0
/1
02

.0
/1
02
н
Gregularization_losses
bmetrics
H	variables
clayer_metrics
dlayer_regularization_losses
Itrainable_variables
enon_trainable_variables

flayers
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
~|
VARIABLE_VALUEAdam/dense_10/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_20/lstm_cell_20/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE,Adam/lstm_20/lstm_cell_20/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_20/lstm_cell_20/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_21/lstm_cell_21/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE,Adam/lstm_21/lstm_cell_21/recurrent_kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_21/lstm_cell_21/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_10/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_10/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_20/lstm_cell_20/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE,Adam/lstm_20/lstm_cell_20/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_20/lstm_cell_20/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/lstm_21/lstm_cell_21/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE,Adam/lstm_21/lstm_cell_21/recurrent_kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/lstm_21/lstm_cell_21/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
И
serving_default_lstm_20_inputPlaceholder*+
_output_shapes
:         ]*
dtype0* 
shape:         ]
м
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_20_inputlstm_20/lstm_cell_20/kernel%lstm_20/lstm_cell_20/recurrent_kernellstm_20/lstm_cell_20/biaslstm_21/lstm_cell_21/kernel%lstm_21/lstm_cell_21/recurrent_kernellstm_21/lstm_cell_21/biasdense_10/kerneldense_10/bias*
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
&__inference_signature_wrapper_39102844
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┴
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
GPU 2J 8В **
f%R#
!__inference__traced_save_39105262
и	
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_39105371АЎ$
у
═
while_cond_39103620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39103620___redundant_placeholder06
2while_while_cond_39103620___redundant_placeholder16
2while_while_cond_39103620___redundant_placeholder26
2while_while_cond_39103620___redundant_placeholder3
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_39101339
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39101339___redundant_placeholder06
2while_while_cond_39101339___redundant_placeholder16
2while_while_cond_39101339___redundant_placeholder26
2while_while_cond_39101339___redundant_placeholder3
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
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
ї
Е
)sequential_10_lstm_21_while_cond_39100508H
Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counterN
Jsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations+
'sequential_10_lstm_21_while_placeholder-
)sequential_10_lstm_21_while_placeholder_1-
)sequential_10_lstm_21_while_placeholder_2-
)sequential_10_lstm_21_while_placeholder_3J
Fsequential_10_lstm_21_while_less_sequential_10_lstm_21_strided_slice_1b
^sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_39100508___redundant_placeholder0b
^sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_39100508___redundant_placeholder1b
^sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_39100508___redundant_placeholder2b
^sequential_10_lstm_21_while_sequential_10_lstm_21_while_cond_39100508___redundant_placeholder3(
$sequential_10_lstm_21_while_identity
▐
 sequential_10/lstm_21/while/LessLess'sequential_10_lstm_21_while_placeholderFsequential_10_lstm_21_while_less_sequential_10_lstm_21_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_10/lstm_21/while/LessЯ
$sequential_10/lstm_21/while/IdentityIdentity$sequential_10/lstm_21/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_10/lstm_21/while/Identity"U
$sequential_10_lstm_21_while_identity-sequential_10/lstm_21/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_39101954
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]°G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°A
2while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив)while/lstm_cell_20/BiasAdd/ReadVariableOpв(while/lstm_cell_20/MatMul/ReadVariableOpв*while/lstm_cell_20/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp╫
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul╨
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOp└
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul_1╕
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/add╚
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOp┼
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/BiasAddК
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dimП
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
while/lstm_cell_20/splitЩ
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/SigmoidЭ
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_1б
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mulР
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu╡
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_1к
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/add_1Э
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_2П
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu_1╣
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2V
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_39102583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102583___redundant_placeholder06
2while_while_cond_39102583___redundant_placeholder16
2while_while_cond_39102583___redundant_placeholder26
2while_while_cond_39102583___redundant_placeholder3
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_39100709
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39100709___redundant_placeholder06
2while_while_cond_39100709___redundant_placeholder16
2while_while_cond_39100709___redundant_placeholder26
2while_while_cond_39100709___redundant_placeholder3
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
Е
f
H__inference_dropout_21_layer_call_and_return_conditional_losses_39104882

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         `2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         `2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
Ю?
╘
while_body_39104296
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮАH
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ЮАF
3while_lstm_cell_21_matmul_1_readvariableop_resource:	`АA
2while_lstm_cell_21_biasadd_readvariableop_resource:	АИв)while/lstm_cell_21/BiasAdd/ReadVariableOpв(while/lstm_cell_21/MatMul/ReadVariableOpв*while/lstm_cell_21/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp╫
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul╧
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOp└
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul_1╕
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/add╚
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOp┼
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/BiasAddК
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dimЛ
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
while/lstm_cell_21/splitШ
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/SigmoidЬ
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_1а
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mulП
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu┤
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_1й
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/add_1Ь
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_2О
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu_1╕
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2V
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
Б
Й
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105074

inputs
states_0
states_12
matmul_readvariableop_resource:
ЮА3
 matmul_1_readvariableop_resource:	`А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
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
L:         `:         `:         `:         `*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         `2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         `2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         `2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         `2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         `2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         `2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         `2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         `2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         `2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         `2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         `2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         `2

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
@:         Ю:         `:         `: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         Ю
 
_user_specified_nameinputs:QM
'
_output_shapes
:         `
"
_user_specified_name
states/0:QM
'
_output_shapes
:         `
"
_user_specified_name
states/1
▀
═
while_cond_39104446
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104446___redundant_placeholder06
2while_while_cond_39104446___redundant_placeholder16
2while_while_cond_39104446___redundant_placeholder26
2while_while_cond_39104446___redundant_placeholder3
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
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_39104748
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104748___redundant_placeholder06
2while_while_cond_39104748___redundant_placeholder16
2while_while_cond_39104748___redundant_placeholder26
2while_while_cond_39104748___redundant_placeholder3
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
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_39104074
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]°G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°A
2while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив)while/lstm_cell_20/BiasAdd/ReadVariableOpв(while/lstm_cell_20/MatMul/ReadVariableOpв*while/lstm_cell_20/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp╫
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul╨
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOp└
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul_1╕
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/add╚
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOp┼
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/BiasAddК
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dimП
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
while/lstm_cell_20/splitЩ
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/SigmoidЭ
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_1б
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mulР
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu╡
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_1к
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/add_1Э
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_2П
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu_1╣
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2V
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
м\
а
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104380
inputs_0?
+lstm_cell_21_matmul_readvariableop_resource:
ЮА@
-lstm_cell_21_matmul_1_readvariableop_resource:	`А;
,lstm_cell_21_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_21/BiasAdd/ReadVariableOpв"lstm_cell_21/MatMul/ReadVariableOpв$lstm_cell_21/MatMul_1/ReadVariableOpвwhileF
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
value	B :`2
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
:         `2
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
value	B :`2
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
:         `2	
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
!:                  Ю2
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
valueB"    Ю   27
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
:         Ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOpн
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul╗
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOpй
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul_1а
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/add┤
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOpн
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dimє
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
lstm_cell_21/splitЖ
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/SigmoidК
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_1Л
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul}
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
lstm_cell_21/ReluЬ
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_1С
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/add_1К
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_2|
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/Relu_1а
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104296*
condR
while_cond_39104295*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
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
:         `*
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
 :                  `2
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
 :                  `2

Identity╚
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Ю: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  Ю
"
_user_specified_name
inputs/0
╧
g
H__inference_dropout_21_layer_call_and_return_conditional_losses_39102305

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
:         `2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         `*
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
:         `2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         `2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         `2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         `2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
Е&
є
while_body_39100920
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_20_39100944_0:	]°1
while_lstm_cell_20_39100946_0:
Ю°,
while_lstm_cell_20_39100948_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_20_39100944:	]°/
while_lstm_cell_20_39100946:
Ю°*
while_lstm_cell_20_39100948:	°Ив*while/lstm_cell_20/StatefulPartitionedCall├
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
*while/lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_20_39100944_0while_lstm_cell_20_39100946_0while_lstm_cell_20_39100948_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ю:         Ю:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391008422,
*while/lstm_cell_20/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_20/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_5З

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
while_lstm_cell_20_39100944while_lstm_cell_20_39100944_0"<
while_lstm_cell_20_39100946while_lstm_cell_20_39100946_0"<
while_lstm_cell_20_39100948while_lstm_cell_20_39100948_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2X
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_39103772
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]°G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°A
2while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив)while/lstm_cell_20/BiasAdd/ReadVariableOpв(while/lstm_cell_20/MatMul/ReadVariableOpв*while/lstm_cell_20/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp╫
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul╨
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOp└
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul_1╕
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/add╚
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOp┼
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/BiasAddК
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dimП
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
while/lstm_cell_20/splitЩ
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/SigmoidЭ
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_1б
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mulР
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu╡
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_1к
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/add_1Э
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_2П
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu_1╣
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2V
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
╛F
О
E__inference_lstm_21_layer_call_and_return_conditional_losses_39101409

inputs)
lstm_cell_21_39101327:
ЮА(
lstm_cell_21_39101329:	`А$
lstm_cell_21_39101331:	А
identityИв$lstm_cell_21/StatefulPartitionedCallвwhileD
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
value	B :`2
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
:         `2
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
value	B :`2
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
:         `2	
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
!:                  Ю2
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
valueB"    Ю   27
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
:         Ю*
shrink_axis_mask2
strided_slice_2е
$lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_21_39101327lstm_cell_21_39101329lstm_cell_21_39101331*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         `:         `:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391013262&
$lstm_cell_21/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_21_39101327lstm_cell_21_39101329lstm_cell_21_39101331*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39101340*
condR
while_cond_39101339*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
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
:         `*
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
 :                  `2
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
 :                  `2

Identity}
NoOpNoOp%^lstm_cell_21/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Ю: : : 2L
$lstm_cell_21/StatefulPartitionedCall$lstm_cell_21/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  Ю
 
_user_specified_nameinputs
▀
═
while_cond_39102387
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102387___redundant_placeholder06
2while_while_cond_39102387___redundant_placeholder16
2while_while_cond_39102387___redundant_placeholder26
2while_while_cond_39102387___redundant_placeholder3
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
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
▀
═
while_cond_39104597
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104597___redundant_placeholder06
2while_while_cond_39104597___redundant_placeholder16
2while_while_cond_39104597___redundant_placeholder26
2while_while_cond_39104597___redundant_placeholder3
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
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
╢
╕
*__inference_lstm_20_layer_call_fn_39104191

inputs
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391020382
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ю2

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
╟
∙
/__inference_lstm_cell_20_layer_call_fn_39105025

inputs
states_0
states_1
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
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
<:         Ю:         Ю:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391006962
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ю2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Ю2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         Ю2

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
A:         ]:         Ю:         Ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ю
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ю
"
_user_specified_name
states/1
╜
∙
/__inference_lstm_cell_21_layer_call_fn_39105123

inputs
states_0
states_1
unknown:
ЮА
	unknown_0:	`А
	unknown_1:	А
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
9:         `:         `:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391013262
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `2

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
@:         Ю:         `:         `: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ю
 
_user_specified_nameinputs:QM
'
_output_shapes
:         `
"
_user_specified_name
states/0:QM
'
_output_shapes
:         `
"
_user_specified_name
states/1
▀
═
while_cond_39102118
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39102118___redundant_placeholder06
2while_while_cond_39102118___redundant_placeholder16
2while_while_cond_39102118___redundant_placeholder26
2while_while_cond_39102118___redundant_placeholder3
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
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
·	
╚
&__inference_signature_wrapper_39102844
lstm_20_input
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
	unknown_2:
ЮА
	unknown_3:	`А
	unknown_4:	А
	unknown_5:`
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_391006212
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
_user_specified_namelstm_20_input
ф^
Ц
)sequential_10_lstm_20_while_body_39100361H
Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counterN
Jsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations+
'sequential_10_lstm_20_while_placeholder-
)sequential_10_lstm_20_while_placeholder_1-
)sequential_10_lstm_20_while_placeholder_2-
)sequential_10_lstm_20_while_placeholder_3G
Csequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1_0Г
sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0\
Isequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	]°_
Ksequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°Y
Jsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	°(
$sequential_10_lstm_20_while_identity*
&sequential_10_lstm_20_while_identity_1*
&sequential_10_lstm_20_while_identity_2*
&sequential_10_lstm_20_while_identity_3*
&sequential_10_lstm_20_while_identity_4*
&sequential_10_lstm_20_while_identity_5E
Asequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1Б
}sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensorZ
Gsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	]°]
Isequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°W
Hsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpв>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpв@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpя
Msequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2O
Msequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape╫
?sequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0'sequential_10_lstm_20_while_placeholderVsequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02A
?sequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItemЛ
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOpIsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02@
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpп
/sequential_10/lstm_20/while/lstm_cell_20/MatMulMatMulFsequential_10/lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °21
/sequential_10/lstm_20/while/lstm_cell_20/MatMulТ
@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOpKsequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype02B
@sequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpШ
1sequential_10/lstm_20/while/lstm_cell_20/MatMul_1MatMul)sequential_10_lstm_20_while_placeholder_2Hsequential_10/lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °23
1sequential_10/lstm_20/while/lstm_cell_20/MatMul_1Р
,sequential_10/lstm_20/while/lstm_cell_20/addAddV29sequential_10/lstm_20/while/lstm_cell_20/MatMul:product:0;sequential_10/lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2.
,sequential_10/lstm_20/while/lstm_cell_20/addК
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOpJsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02A
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpЭ
0sequential_10/lstm_20/while/lstm_cell_20/BiasAddBiasAdd0sequential_10/lstm_20/while/lstm_cell_20/add:z:0Gsequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °22
0sequential_10/lstm_20/while/lstm_cell_20/BiasAdd╢
8sequential_10/lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_10/lstm_20/while/lstm_cell_20/split/split_dimч
.sequential_10/lstm_20/while/lstm_cell_20/splitSplitAsequential_10/lstm_20/while/lstm_cell_20/split/split_dim:output:09sequential_10/lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split20
.sequential_10/lstm_20/while/lstm_cell_20/split█
0sequential_10/lstm_20/while/lstm_cell_20/SigmoidSigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю22
0sequential_10/lstm_20/while/lstm_cell_20/Sigmoid▀
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю24
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1∙
,sequential_10/lstm_20/while/lstm_cell_20/mulMul6sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_1:y:0)sequential_10_lstm_20_while_placeholder_3*
T0*(
_output_shapes
:         Ю2.
,sequential_10/lstm_20/while/lstm_cell_20/mul╥
-sequential_10/lstm_20/while/lstm_cell_20/ReluRelu7sequential_10/lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2/
-sequential_10/lstm_20/while/lstm_cell_20/ReluН
.sequential_10/lstm_20/while/lstm_cell_20/mul_1Mul4sequential_10/lstm_20/while/lstm_cell_20/Sigmoid:y:0;sequential_10/lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю20
.sequential_10/lstm_20/while/lstm_cell_20/mul_1В
.sequential_10/lstm_20/while/lstm_cell_20/add_1AddV20sequential_10/lstm_20/while/lstm_cell_20/mul:z:02sequential_10/lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю20
.sequential_10/lstm_20/while/lstm_cell_20/add_1▀
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid7sequential_10/lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю24
2sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2╤
/sequential_10/lstm_20/while/lstm_cell_20/Relu_1Relu2sequential_10/lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю21
/sequential_10/lstm_20/while/lstm_cell_20/Relu_1С
.sequential_10/lstm_20/while/lstm_cell_20/mul_2Mul6sequential_10/lstm_20/while/lstm_cell_20/Sigmoid_2:y:0=sequential_10/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю20
.sequential_10/lstm_20/while/lstm_cell_20/mul_2╬
@sequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_10_lstm_20_while_placeholder_1'sequential_10_lstm_20_while_placeholder2sequential_10/lstm_20/while/lstm_cell_20/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_10/lstm_20/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_10/lstm_20/while/add/y┴
sequential_10/lstm_20/while/addAddV2'sequential_10_lstm_20_while_placeholder*sequential_10/lstm_20/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_20/while/addМ
#sequential_10/lstm_20/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_10/lstm_20/while/add_1/yф
!sequential_10/lstm_20/while/add_1AddV2Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counter,sequential_10/lstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_20/while/add_1├
$sequential_10/lstm_20/while/IdentityIdentity%sequential_10/lstm_20/while/add_1:z:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_10/lstm_20/while/Identityь
&sequential_10/lstm_20/while/Identity_1IdentityJsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_1┼
&sequential_10/lstm_20/while/Identity_2Identity#sequential_10/lstm_20/while/add:z:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_2Є
&sequential_10/lstm_20/while/Identity_3IdentityPsequential_10/lstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_10/lstm_20/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_20/while/Identity_3ц
&sequential_10/lstm_20/while/Identity_4Identity2sequential_10/lstm_20/while/lstm_cell_20/mul_2:z:0!^sequential_10/lstm_20/while/NoOp*
T0*(
_output_shapes
:         Ю2(
&sequential_10/lstm_20/while/Identity_4ц
&sequential_10/lstm_20/while/Identity_5Identity2sequential_10/lstm_20/while/lstm_cell_20/add_1:z:0!^sequential_10/lstm_20/while/NoOp*
T0*(
_output_shapes
:         Ю2(
&sequential_10/lstm_20/while/Identity_5╠
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
&sequential_10_lstm_20_while_identity_5/sequential_10/lstm_20/while/Identity_5:output:0"Ц
Hsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resourceJsequential_10_lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0"Ш
Isequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resourceKsequential_10_lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0"Ф
Gsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resourceIsequential_10_lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"И
Asequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1Csequential_10_lstm_20_while_sequential_10_lstm_20_strided_slice_1_0"А
}sequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensorsequential_10_lstm_20_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2В
?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp?sequential_10/lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp2А
>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp>sequential_10/lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp2Д
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
Ч

╦
0__inference_sequential_10_layer_call_fn_39103533

inputs
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
	unknown_2:
ЮА
	unknown_3:	`А
	unknown_4:	А
	unknown_5:`
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_391022562
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
▀
═
while_cond_39104295
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104295___redundant_placeholder06
2while_while_cond_39104295___redundant_placeholder16
2while_while_cond_39104295___redundant_placeholder26
2while_while_cond_39104295___redundant_placeholder3
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
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_39102584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]°G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°A
2while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив)while/lstm_cell_20/BiasAdd/ReadVariableOpв(while/lstm_cell_20/MatMul/ReadVariableOpв*while/lstm_cell_20/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp╫
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul╨
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOp└
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul_1╕
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/add╚
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOp┼
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/BiasAddК
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dimП
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
while/lstm_cell_20/splitЩ
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/SigmoidЭ
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_1б
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mulР
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu╡
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_1к
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/add_1Э
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_2П
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu_1╣
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2V
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
¤%
є
while_body_39101340
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_21_39101364_0:
ЮА0
while_lstm_cell_21_39101366_0:	`А,
while_lstm_cell_21_39101368_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_21_39101364:
ЮА.
while_lstm_cell_21_39101366:	`А*
while_lstm_cell_21_39101368:	АИв*while/lstm_cell_21/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemщ
*while/lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_21_39101364_0while_lstm_cell_21_39101366_0while_lstm_cell_21_39101368_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         `:         `:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391013262,
*while/lstm_cell_21/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3д
while/Identity_4Identity3while/lstm_cell_21/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_21/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_5З

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
while_lstm_cell_21_39101364while_lstm_cell_21_39101364_0"<
while_lstm_cell_21_39101366while_lstm_cell_21_39101366_0"<
while_lstm_cell_21_39101368while_lstm_cell_21_39101368_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2X
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
Л
З
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39100696

inputs

states
states_11
matmul_readvariableop_resource:	]°4
 matmul_1_readvariableop_resource:
Ю°.
biasadd_readvariableop_resource:	°
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         °2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2	
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
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ю2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ю2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ю2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ю2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ю2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ю2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ю2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ю2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

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
A:         ]:         Ю:         Ю: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Ю
 
_user_specified_namestates:PL
(
_output_shapes
:         Ю
 
_user_specified_namestates
╢
f
-__inference_dropout_21_layer_call_fn_39104904

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
:         `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391023052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
╦F
О
E__inference_lstm_20_layer_call_and_return_conditional_losses_39100989

inputs(
lstm_cell_20_39100907:	]°)
lstm_cell_20_39100909:
Ю°$
lstm_cell_20_39100911:	°
identityИв$lstm_cell_20/StatefulPartitionedCallвwhileD
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
B :Ю2
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
B :Ю2
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
:         Ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
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
B :Ю2
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
:         Ю2	
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
$lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_20_39100907lstm_cell_20_39100909lstm_cell_20_39100911*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ю:         Ю:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391008422&
$lstm_cell_20/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_20_39100907lstm_cell_20_39100909lstm_cell_20_39100911*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39100920*
condR
while_cond_39100919*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ю*
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
:         Ю*
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
!:                  Ю2
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
!:                  Ю2

Identity}
NoOpNoOp%^lstm_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_20/StatefulPartitionedCall$lstm_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
ПK
▄
!__inference__traced_save_39105262
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
ShardedFilename╕
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*╩
value└B╜"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╠
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┼
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_10_kernel_read_readvariableop(savev2_dense_10_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_20_lstm_cell_20_kernel_read_readvariableop@savev2_lstm_20_lstm_cell_20_recurrent_kernel_read_readvariableop4savev2_lstm_20_lstm_cell_20_bias_read_readvariableop6savev2_lstm_21_lstm_cell_21_kernel_read_readvariableop@savev2_lstm_21_lstm_cell_21_recurrent_kernel_read_readvariableop4savev2_lstm_21_lstm_cell_21_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_10_kernel_m_read_readvariableop/savev2_adam_dense_10_bias_m_read_readvariableop=savev2_adam_lstm_20_lstm_cell_20_kernel_m_read_readvariableopGsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_20_lstm_cell_20_bias_m_read_readvariableop=savev2_adam_lstm_21_lstm_cell_21_kernel_m_read_readvariableopGsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_21_lstm_cell_21_bias_m_read_readvariableop1savev2_adam_dense_10_kernel_v_read_readvariableop/savev2_adam_dense_10_bias_v_read_readvariableop=savev2_adam_lstm_20_lstm_cell_20_kernel_v_read_readvariableopGsavev2_adam_lstm_20_lstm_cell_20_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_20_lstm_cell_20_bias_v_read_readvariableop=savev2_adam_lstm_21_lstm_cell_21_kernel_v_read_readvariableopGsavev2_adam_lstm_21_lstm_cell_21_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_21_lstm_cell_21_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
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

identity_1Identity_1:output:0*П
_input_shapes¤
·: :`:: : : : : :	]°:
Ю°:°:
ЮА:	`А:А: : : : :`::	]°:
Ю°:°:
ЮА:	`А:А:`::	]°:
Ю°:°:
ЮА:	`А:А: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:`: 
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
:	]°:&	"
 
_output_shapes
:
Ю°:!


_output_shapes	
:°:&"
 
_output_shapes
:
ЮА:%!

_output_shapes
:	`А:!

_output_shapes	
:А:
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

:`: 

_output_shapes
::%!

_output_shapes
:	]°:&"
 
_output_shapes
:
Ю°:!

_output_shapes	
:°:&"
 
_output_shapes
:
ЮА:%!

_output_shapes
:	`А:!

_output_shapes	
:А:$ 

_output_shapes

:`: 

_output_shapes
::%!

_output_shapes
:	]°:&"
 
_output_shapes
:
Ю°:!

_output_shapes	
:°:&"
 
_output_shapes
:
ЮА:% !

_output_shapes
:	`А:!!

_output_shapes	
:А:"

_output_shapes
: 
Й
f
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104207

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ю2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ю2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ю:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
╘

э
lstm_20_while_cond_39103237,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3.
*lstm_20_while_less_lstm_20_strided_slice_1F
Blstm_20_while_lstm_20_while_cond_39103237___redundant_placeholder0F
Blstm_20_while_lstm_20_while_cond_39103237___redundant_placeholder1F
Blstm_20_while_lstm_20_while_cond_39103237___redundant_placeholder2F
Blstm_20_while_lstm_20_while_cond_39103237___redundant_placeholder3
lstm_20_while_identity
Ш
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
╬М
К
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103512

inputsF
3lstm_20_lstm_cell_20_matmul_readvariableop_resource:	]°I
5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°C
4lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	°G
3lstm_21_lstm_cell_21_matmul_readvariableop_resource:
ЮАH
5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:	`АC
4lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	А<
*dense_10_tensordot_readvariableop_resource:`6
(dense_10_biasadd_readvariableop_resource:
identityИвdense_10/BiasAdd/ReadVariableOpв!dense_10/Tensordot/ReadVariableOpв+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpв*lstm_20/lstm_cell_20/MatMul/ReadVariableOpв,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpвlstm_20/whileв+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpв*lstm_21/lstm_cell_21/MatMul/ReadVariableOpв,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpвlstm_21/whileT
lstm_20/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_20/ShapeД
lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice/stackИ
lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_1И
lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_2Т
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
B :Ю2
lstm_20/zeros/mul/yМ
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
B :ш2
lstm_20/zeros/Less/yЗ
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
B :Ю2
lstm_20/zeros/packed/1г
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
lstm_20/zeros/ConstЦ
lstm_20/zerosFilllstm_20/zeros/packed:output:0lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:         Ю2
lstm_20/zerosq
lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
lstm_20/zeros_1/mul/yТ
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
B :ш2
lstm_20/zeros_1/Less/yП
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
B :Ю2
lstm_20/zeros_1/packed/1й
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
lstm_20/zeros_1/ConstЮ
lstm_20/zeros_1Filllstm_20/zeros_1/packed:output:0lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ю2
lstm_20/zeros_1Е
lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose/permТ
lstm_20/transpose	Transposeinputslstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_20/transposeg
lstm_20/Shape_1Shapelstm_20/transpose:y:0*
T0*
_output_shapes
:2
lstm_20/Shape_1И
lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_1/stackМ
lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_1М
lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_2Ю
lstm_20/strided_slice_1StridedSlicelstm_20/Shape_1:output:0&lstm_20/strided_slice_1/stack:output:0(lstm_20/strided_slice_1/stack_1:output:0(lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slice_1Х
#lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_20/TensorArrayV2/element_shape╥
lstm_20/TensorArrayV2TensorListReserve,lstm_20/TensorArrayV2/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2╧
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_20/transpose:y:0Flstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_20/TensorArrayUnstack/TensorListFromTensorИ
lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_2/stackМ
lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_1М
lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_2м
lstm_20/strided_slice_2StridedSlicelstm_20/transpose:y:0&lstm_20/strided_slice_2/stack:output:0(lstm_20/strided_slice_2/stack_1:output:0(lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_20/strided_slice_2═
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02,
*lstm_20/lstm_cell_20/MatMul/ReadVariableOp═
lstm_20/lstm_cell_20/MatMulMatMul lstm_20/strided_slice_2:output:02lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_20/lstm_cell_20/MatMul╘
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02.
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp╔
lstm_20/lstm_cell_20/MatMul_1MatMullstm_20/zeros:output:04lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_20/lstm_cell_20/MatMul_1└
lstm_20/lstm_cell_20/addAddV2%lstm_20/lstm_cell_20/MatMul:product:0'lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
lstm_20/lstm_cell_20/add╠
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02-
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp═
lstm_20/lstm_cell_20/BiasAddBiasAddlstm_20/lstm_cell_20/add:z:03lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_20/lstm_cell_20/BiasAddО
$lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_20/lstm_cell_20/split/split_dimЧ
lstm_20/lstm_cell_20/splitSplit-lstm_20/lstm_cell_20/split/split_dim:output:0%lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
lstm_20/lstm_cell_20/splitЯ
lstm_20/lstm_cell_20/SigmoidSigmoid#lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/Sigmoidг
lstm_20/lstm_cell_20/Sigmoid_1Sigmoid#lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2 
lstm_20/lstm_cell_20/Sigmoid_1м
lstm_20/lstm_cell_20/mulMul"lstm_20/lstm_cell_20/Sigmoid_1:y:0lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/mulЦ
lstm_20/lstm_cell_20/ReluRelu#lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/Relu╜
lstm_20/lstm_cell_20/mul_1Mul lstm_20/lstm_cell_20/Sigmoid:y:0'lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/mul_1▓
lstm_20/lstm_cell_20/add_1AddV2lstm_20/lstm_cell_20/mul:z:0lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/add_1г
lstm_20/lstm_cell_20/Sigmoid_2Sigmoid#lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2 
lstm_20/lstm_cell_20/Sigmoid_2Х
lstm_20/lstm_cell_20/Relu_1Relulstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/Relu_1┴
lstm_20/lstm_cell_20/mul_2Mul"lstm_20/lstm_cell_20/Sigmoid_2:y:0)lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/mul_2Я
%lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2'
%lstm_20/TensorArrayV2_1/element_shape╪
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
lstm_20/timeП
 lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_20/while/maximum_iterationsz
lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/while/loop_counterЛ
lstm_20/whileWhile#lstm_20/while/loop_counter:output:0)lstm_20/while/maximum_iterations:output:0lstm_20/time:output:0 lstm_20/TensorArrayV2_1:handle:0lstm_20/zeros:output:0lstm_20/zeros_1:output:0 lstm_20/strided_slice_1:output:0?lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_20_lstm_cell_20_matmul_readvariableop_resource5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_20_while_body_39103238*'
condR
lstm_20_while_cond_39103237*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
lstm_20/while┼
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2:
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_20/TensorArrayV2Stack/TensorListStackTensorListStacklstm_20/while:output:3Alstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ю*
element_dtype02,
*lstm_20/TensorArrayV2Stack/TensorListStackС
lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_20/strided_slice_3/stackМ
lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_20/strided_slice_3/stack_1М
lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_3/stack_2╦
lstm_20/strided_slice_3StridedSlice3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_20/strided_slice_3/stack:output:0(lstm_20/strided_slice_3/stack_1:output:0(lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ю*
shrink_axis_mask2
lstm_20/strided_slice_3Й
lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose_1/perm╞
lstm_20/transpose_1	Transpose3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ю2
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
 *лкк?2
dropout_20/dropout/Constк
dropout_20/dropout/MulMullstm_20/transpose_1:y:0!dropout_20/dropout/Const:output:0*
T0*,
_output_shapes
:         Ю2
dropout_20/dropout/Mul{
dropout_20/dropout/ShapeShapelstm_20/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_20/dropout/Shape┌
/dropout_20/dropout/random_uniform/RandomUniformRandomUniform!dropout_20/dropout/Shape:output:0*
T0*,
_output_shapes
:         Ю*
dtype021
/dropout_20/dropout/random_uniform/RandomUniformЛ
!dropout_20/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_20/dropout/GreaterEqual/yя
dropout_20/dropout/GreaterEqualGreaterEqual8dropout_20/dropout/random_uniform/RandomUniform:output:0*dropout_20/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         Ю2!
dropout_20/dropout/GreaterEqualе
dropout_20/dropout/CastCast#dropout_20/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ю2
dropout_20/dropout/Castл
dropout_20/dropout/Mul_1Muldropout_20/dropout/Mul:z:0dropout_20/dropout/Cast:y:0*
T0*,
_output_shapes
:         Ю2
dropout_20/dropout/Mul_1j
lstm_21/ShapeShapedropout_20/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_21/ShapeД
lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice/stackИ
lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_1И
lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_2Т
lstm_21/strided_sliceStridedSlicelstm_21/Shape:output:0$lstm_21/strided_slice/stack:output:0&lstm_21/strided_slice/stack_1:output:0&lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slicel
lstm_21/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
lstm_21/zeros/mul/yМ
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
B :ш2
lstm_21/zeros/Less/yЗ
lstm_21/zeros/LessLesslstm_21/zeros/mul:z:0lstm_21/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros/Lessr
lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2
lstm_21/zeros/packed/1г
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
lstm_21/zeros/ConstХ
lstm_21/zerosFilllstm_21/zeros/packed:output:0lstm_21/zeros/Const:output:0*
T0*'
_output_shapes
:         `2
lstm_21/zerosp
lstm_21/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
lstm_21/zeros_1/mul/yТ
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
B :ш2
lstm_21/zeros_1/Less/yП
lstm_21/zeros_1/LessLesslstm_21/zeros_1/mul:z:0lstm_21/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros_1/Lessv
lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2
lstm_21/zeros_1/packed/1й
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
lstm_21/zeros_1/ConstЭ
lstm_21/zeros_1Filllstm_21/zeros_1/packed:output:0lstm_21/zeros_1/Const:output:0*
T0*'
_output_shapes
:         `2
lstm_21/zeros_1Е
lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose/permй
lstm_21/transpose	Transposedropout_20/dropout/Mul_1:z:0lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:         Ю2
lstm_21/transposeg
lstm_21/Shape_1Shapelstm_21/transpose:y:0*
T0*
_output_shapes
:2
lstm_21/Shape_1И
lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_1/stackМ
lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_1М
lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_2Ю
lstm_21/strided_slice_1StridedSlicelstm_21/Shape_1:output:0&lstm_21/strided_slice_1/stack:output:0(lstm_21/strided_slice_1/stack_1:output:0(lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slice_1Х
#lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_21/TensorArrayV2/element_shape╥
lstm_21/TensorArrayV2TensorListReserve,lstm_21/TensorArrayV2/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2╧
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2?
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_21/transpose:y:0Flstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_21/TensorArrayUnstack/TensorListFromTensorИ
lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_2/stackМ
lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_1М
lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_2н
lstm_21/strided_slice_2StridedSlicelstm_21/transpose:y:0&lstm_21/strided_slice_2/stack:output:0(lstm_21/strided_slice_2/stack_1:output:0(lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ю*
shrink_axis_mask2
lstm_21/strided_slice_2╬
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02,
*lstm_21/lstm_cell_21/MatMul/ReadVariableOp═
lstm_21/lstm_cell_21/MatMulMatMul lstm_21/strided_slice_2:output:02lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_21/lstm_cell_21/MatMul╙
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02.
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp╔
lstm_21/lstm_cell_21/MatMul_1MatMullstm_21/zeros:output:04lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_21/lstm_cell_21/MatMul_1└
lstm_21/lstm_cell_21/addAddV2%lstm_21/lstm_cell_21/MatMul:product:0'lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_21/lstm_cell_21/add╠
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp═
lstm_21/lstm_cell_21/BiasAddBiasAddlstm_21/lstm_cell_21/add:z:03lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_21/lstm_cell_21/BiasAddО
$lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_21/lstm_cell_21/split/split_dimУ
lstm_21/lstm_cell_21/splitSplit-lstm_21/lstm_cell_21/split/split_dim:output:0%lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
lstm_21/lstm_cell_21/splitЮ
lstm_21/lstm_cell_21/SigmoidSigmoid#lstm_21/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/Sigmoidв
lstm_21/lstm_cell_21/Sigmoid_1Sigmoid#lstm_21/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2 
lstm_21/lstm_cell_21/Sigmoid_1л
lstm_21/lstm_cell_21/mulMul"lstm_21/lstm_cell_21/Sigmoid_1:y:0lstm_21/zeros_1:output:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/mulХ
lstm_21/lstm_cell_21/ReluRelu#lstm_21/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/Relu╝
lstm_21/lstm_cell_21/mul_1Mul lstm_21/lstm_cell_21/Sigmoid:y:0'lstm_21/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/mul_1▒
lstm_21/lstm_cell_21/add_1AddV2lstm_21/lstm_cell_21/mul:z:0lstm_21/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/add_1в
lstm_21/lstm_cell_21/Sigmoid_2Sigmoid#lstm_21/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2 
lstm_21/lstm_cell_21/Sigmoid_2Ф
lstm_21/lstm_cell_21/Relu_1Relulstm_21/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/Relu_1└
lstm_21/lstm_cell_21/mul_2Mul"lstm_21/lstm_cell_21/Sigmoid_2:y:0)lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/mul_2Я
%lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_21/TensorArrayV2_1/element_shape╪
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
lstm_21/timeП
 lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_21/while/maximum_iterationsz
lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/while/loop_counterЗ
lstm_21/whileWhile#lstm_21/while/loop_counter:output:0)lstm_21/while/maximum_iterations:output:0lstm_21/time:output:0 lstm_21/TensorArrayV2_1:handle:0lstm_21/zeros:output:0lstm_21/zeros_1:output:0 lstm_21/strided_slice_1:output:0?lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_21_lstm_cell_21_matmul_readvariableop_resource5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_21_while_body_39103393*'
condR
lstm_21_while_cond_39103392*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
lstm_21/while┼
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2:
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_21/TensorArrayV2Stack/TensorListStackTensorListStacklstm_21/while:output:3Alstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         `*
element_dtype02,
*lstm_21/TensorArrayV2Stack/TensorListStackС
lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_21/strided_slice_3/stackМ
lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_21/strided_slice_3/stack_1М
lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_3/stack_2╩
lstm_21/strided_slice_3StridedSlice3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_21/strided_slice_3/stack:output:0(lstm_21/strided_slice_3/stack_1:output:0(lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         `*
shrink_axis_mask2
lstm_21/strided_slice_3Й
lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose_1/perm┼
lstm_21/transpose_1	Transpose3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_21/transpose_1/perm:output:0*
T0*+
_output_shapes
:         `2
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
 *  а?2
dropout_21/dropout/Constй
dropout_21/dropout/MulMullstm_21/transpose_1:y:0!dropout_21/dropout/Const:output:0*
T0*+
_output_shapes
:         `2
dropout_21/dropout/Mul{
dropout_21/dropout/ShapeShapelstm_21/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_21/dropout/Shape┘
/dropout_21/dropout/random_uniform/RandomUniformRandomUniform!dropout_21/dropout/Shape:output:0*
T0*+
_output_shapes
:         `*
dtype021
/dropout_21/dropout/random_uniform/RandomUniformЛ
!dropout_21/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_21/dropout/GreaterEqual/yю
dropout_21/dropout/GreaterEqualGreaterEqual8dropout_21/dropout/random_uniform/RandomUniform:output:0*dropout_21/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:         `2!
dropout_21/dropout/GreaterEqualд
dropout_21/dropout/CastCast#dropout_21/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         `2
dropout_21/dropout/Castк
dropout_21/dropout/Mul_1Muldropout_21/dropout/Mul:z:0dropout_21/dropout/Cast:y:0*
T0*+
_output_shapes
:         `2
dropout_21/dropout/Mul_1▒
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:`*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axesГ
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/freeА
dense_10/Tensordot/ShapeShapedropout_21/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_10/Tensordot/ShapeЖ
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axis■
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2К
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axisД
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
dense_10/Tensordot/Constд
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/ProdВ
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1м
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1В
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axis▌
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat░
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stack┴
dense_10/Tensordot/transpose	Transposedropout_21/dropout/Mul_1:z:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:         `2
dense_10/Tensordot/transpose├
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_10/Tensordot/Reshape┬
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_10/Tensordot/MatMulВ
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/Const_2Ж
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisъ
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1┤
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_10/Tensordotз
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpл
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_10/BiasAddА
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_10/Softmaxy
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp,^lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+^lstm_20/lstm_cell_20/MatMul/ReadVariableOp-^lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^lstm_20/while,^lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+^lstm_21/lstm_cell_21/MatMul/ReadVariableOp-^lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^lstm_21/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
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
:         ]
 
_user_specified_nameinputs
Е&
є
while_body_39100710
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_20_39100734_0:	]°1
while_lstm_cell_20_39100736_0:
Ю°,
while_lstm_cell_20_39100738_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_20_39100734:	]°/
while_lstm_cell_20_39100736:
Ю°*
while_lstm_cell_20_39100738:	°Ив*while/lstm_cell_20/StatefulPartitionedCall├
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
*while/lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_20_39100734_0while_lstm_cell_20_39100736_0while_lstm_cell_20_39100738_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ю:         Ю:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391006962,
*while/lstm_cell_20/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_20/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_20/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_5З

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
while_lstm_cell_20_39100734while_lstm_cell_20_39100734_0"<
while_lstm_cell_20_39100736while_lstm_cell_20_39100736_0"<
while_lstm_cell_20_39100738while_lstm_cell_20_39100738_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2X
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_39100919
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39100919___redundant_placeholder06
2while_while_cond_39100919___redundant_placeholder16
2while_while_cond_39100919___redundant_placeholder26
2while_while_cond_39100919___redundant_placeholder3
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
╜
∙
/__inference_lstm_cell_21_layer_call_fn_39105140

inputs
states_0
states_1
unknown:
ЮА
	unknown_0:	`А
	unknown_1:	А
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
9:         `:         `:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391014722
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         `2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `2

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
@:         Ю:         `:         `: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ю
 
_user_specified_nameinputs:QM
'
_output_shapes
:         `
"
_user_specified_name
states/0:QM
'
_output_shapes
:         `
"
_user_specified_name
states/1
Й
f
H__inference_dropout_20_layer_call_and_return_conditional_losses_39102051

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         Ю2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         Ю2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ю:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
╦F
О
E__inference_lstm_20_layer_call_and_return_conditional_losses_39100779

inputs(
lstm_cell_20_39100697:	]°)
lstm_cell_20_39100699:
Ю°$
lstm_cell_20_39100701:	°
identityИв$lstm_cell_20/StatefulPartitionedCallвwhileD
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
B :Ю2
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
B :Ю2
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
:         Ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
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
B :Ю2
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
:         Ю2	
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
$lstm_cell_20/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_20_39100697lstm_cell_20_39100699lstm_cell_20_39100701*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         Ю:         Ю:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391006962&
$lstm_cell_20/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_20_39100697lstm_cell_20_39100699lstm_cell_20_39100701*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39100710*
condR
while_cond_39100709*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ю*
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
:         Ю*
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
!:                  Ю2
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
!:                  Ю2

Identity}
NoOpNoOp%^lstm_cell_20/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_20/StatefulPartitionedCall$lstm_cell_20/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
э[
Ю
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104682

inputs?
+lstm_cell_21_matmul_readvariableop_resource:
ЮА@
-lstm_cell_21_matmul_1_readvariableop_resource:	`А;
,lstm_cell_21_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_21/BiasAdd/ReadVariableOpв"lstm_cell_21/MatMul/ReadVariableOpв$lstm_cell_21/MatMul_1/ReadVariableOpвwhileD
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
value	B :`2
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
:         `2
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
value	B :`2
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
:         `2	
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
:         Ю2
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
valueB"    Ю   27
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
:         Ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOpн
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul╗
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOpй
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul_1а
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/add┤
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOpн
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dimє
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
lstm_cell_21/splitЖ
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/SigmoidК
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_1Л
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul}
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
lstm_cell_21/ReluЬ
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_1С
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/add_1К
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_2|
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/Relu_1а
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104598*
condR
while_cond_39104597*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         `*
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
:         `*
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
:         `2
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
:         `2

Identity╚
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Ю: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
├\
а
E__inference_lstm_20_layer_call_and_return_conditional_losses_39103705
inputs_0>
+lstm_cell_20_matmul_readvariableop_resource:	]°A
-lstm_cell_20_matmul_1_readvariableop_resource:
Ю°;
,lstm_cell_20_biasadd_readvariableop_resource:	°
identityИв#lstm_cell_20/BiasAdd/ReadVariableOpв"lstm_cell_20/MatMul/ReadVariableOpв$lstm_cell_20/MatMul_1/ReadVariableOpвwhileF
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
B :Ю2
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
B :Ю2
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
:         Ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
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
B :Ю2
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
:         Ю2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOpн
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul╝
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOpй
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul_1а
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/add┤
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOpн
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dimў
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
lstm_cell_20/splitЗ
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/SigmoidЛ
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_1М
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/ReluЭ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_1Т
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/add_1Л
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Relu_1б
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39103621*
condR
while_cond_39103620*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ю*
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
:         Ю*
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
!:                  Ю2
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
!:                  Ю2

Identity╚
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
╧
g
H__inference_dropout_21_layer_call_and_return_conditional_losses_39104894

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
:         `2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╕
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:         `*
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
:         `2
dropout/GreaterEqualГ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:         `2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:         `2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:         `2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
Д\
Ю
E__inference_lstm_20_layer_call_and_return_conditional_losses_39102668

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	]°A
-lstm_cell_20_matmul_1_readvariableop_resource:
Ю°;
,lstm_cell_20_biasadd_readvariableop_resource:	°
identityИв#lstm_cell_20/BiasAdd/ReadVariableOpв"lstm_cell_20/MatMul/ReadVariableOpв$lstm_cell_20/MatMul_1/ReadVariableOpвwhileD
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
B :Ю2
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
B :Ю2
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
:         Ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
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
B :Ю2
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
:         Ю2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOpн
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul╝
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOpй
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul_1а
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/add┤
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOpн
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dimў
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
lstm_cell_20/splitЗ
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/SigmoidЛ
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_1М
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/ReluЭ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_1Т
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/add_1Л
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Relu_1б
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102584*
condR
while_cond_39102583*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ю*
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
:         Ю*
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
:         Ю2
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
:         Ю2

Identity╚
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
И╣
х	
#__inference__wrapped_model_39100621
lstm_20_inputT
Asequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resource:	]°W
Csequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°Q
Bsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	°U
Asequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resource:
ЮАV
Csequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:	`АQ
Bsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	АJ
8sequential_10_dense_10_tensordot_readvariableop_resource:`D
6sequential_10_dense_10_biasadd_readvariableop_resource:
identityИв-sequential_10/dense_10/BiasAdd/ReadVariableOpв/sequential_10/dense_10/Tensordot/ReadVariableOpв9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpв8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOpв:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpвsequential_10/lstm_20/whileв9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpв8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOpв:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpвsequential_10/lstm_21/whilew
sequential_10/lstm_20/ShapeShapelstm_20_input*
T0*
_output_shapes
:2
sequential_10/lstm_20/Shapeа
)sequential_10/lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_10/lstm_20/strided_slice/stackд
+sequential_10/lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_20/strided_slice/stack_1д
+sequential_10/lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_20/strided_slice/stack_2ц
#sequential_10/lstm_20/strided_sliceStridedSlice$sequential_10/lstm_20/Shape:output:02sequential_10/lstm_20/strided_slice/stack:output:04sequential_10/lstm_20/strided_slice/stack_1:output:04sequential_10/lstm_20/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_10/lstm_20/strided_sliceЙ
!sequential_10/lstm_20/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2#
!sequential_10/lstm_20/zeros/mul/y─
sequential_10/lstm_20/zeros/mulMul,sequential_10/lstm_20/strided_slice:output:0*sequential_10/lstm_20/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_20/zeros/mulЛ
"sequential_10/lstm_20/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_10/lstm_20/zeros/Less/y┐
 sequential_10/lstm_20/zeros/LessLess#sequential_10/lstm_20/zeros/mul:z:0+sequential_10/lstm_20/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_10/lstm_20/zeros/LessП
$sequential_10/lstm_20/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю2&
$sequential_10/lstm_20/zeros/packed/1█
"sequential_10/lstm_20/zeros/packedPack,sequential_10/lstm_20/strided_slice:output:0-sequential_10/lstm_20/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_10/lstm_20/zeros/packedЛ
!sequential_10/lstm_20/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_10/lstm_20/zeros/Const╬
sequential_10/lstm_20/zerosFill+sequential_10/lstm_20/zeros/packed:output:0*sequential_10/lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:         Ю2
sequential_10/lstm_20/zerosН
#sequential_10/lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2%
#sequential_10/lstm_20/zeros_1/mul/y╩
!sequential_10/lstm_20/zeros_1/mulMul,sequential_10/lstm_20/strided_slice:output:0,sequential_10/lstm_20/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_20/zeros_1/mulП
$sequential_10/lstm_20/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_10/lstm_20/zeros_1/Less/y╟
"sequential_10/lstm_20/zeros_1/LessLess%sequential_10/lstm_20/zeros_1/mul:z:0-sequential_10/lstm_20/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_10/lstm_20/zeros_1/LessУ
&sequential_10/lstm_20/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :Ю2(
&sequential_10/lstm_20/zeros_1/packed/1с
$sequential_10/lstm_20/zeros_1/packedPack,sequential_10/lstm_20/strided_slice:output:0/sequential_10/lstm_20/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_10/lstm_20/zeros_1/packedП
#sequential_10/lstm_20/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_10/lstm_20/zeros_1/Const╓
sequential_10/lstm_20/zeros_1Fill-sequential_10/lstm_20/zeros_1/packed:output:0,sequential_10/lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ю2
sequential_10/lstm_20/zeros_1б
$sequential_10/lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_10/lstm_20/transpose/perm├
sequential_10/lstm_20/transpose	Transposelstm_20_input-sequential_10/lstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2!
sequential_10/lstm_20/transposeС
sequential_10/lstm_20/Shape_1Shape#sequential_10/lstm_20/transpose:y:0*
T0*
_output_shapes
:2
sequential_10/lstm_20/Shape_1д
+sequential_10/lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_20/strided_slice_1/stackи
-sequential_10/lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_1/stack_1и
-sequential_10/lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_1/stack_2Є
%sequential_10/lstm_20/strided_slice_1StridedSlice&sequential_10/lstm_20/Shape_1:output:04sequential_10/lstm_20/strided_slice_1/stack:output:06sequential_10/lstm_20/strided_slice_1/stack_1:output:06sequential_10/lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_1▒
1sequential_10/lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_10/lstm_20/TensorArrayV2/element_shapeК
#sequential_10/lstm_20/TensorArrayV2TensorListReserve:sequential_10/lstm_20/TensorArrayV2/element_shape:output:0.sequential_10/lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_10/lstm_20/TensorArrayV2ы
Ksequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2M
Ksequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_10/lstm_20/transpose:y:0Tsequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensorд
+sequential_10/lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_20/strided_slice_2/stackи
-sequential_10/lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_2/stack_1и
-sequential_10/lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_2/stack_2А
%sequential_10/lstm_20/strided_slice_2StridedSlice#sequential_10/lstm_20/transpose:y:04sequential_10/lstm_20/strided_slice_2/stack:output:06sequential_10/lstm_20/strided_slice_2/stack_1:output:06sequential_10/lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_2ў
8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOpAsequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02:
8sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOpЕ
)sequential_10/lstm_20/lstm_cell_20/MatMulMatMul.sequential_10/lstm_20/strided_slice_2:output:0@sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2+
)sequential_10/lstm_20/lstm_cell_20/MatMul■
:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOpCsequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02<
:sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpБ
+sequential_10/lstm_20/lstm_cell_20/MatMul_1MatMul$sequential_10/lstm_20/zeros:output:0Bsequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2-
+sequential_10/lstm_20/lstm_cell_20/MatMul_1°
&sequential_10/lstm_20/lstm_cell_20/addAddV23sequential_10/lstm_20/lstm_cell_20/MatMul:product:05sequential_10/lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2(
&sequential_10/lstm_20/lstm_cell_20/addЎ
9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOpBsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02;
9sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpЕ
*sequential_10/lstm_20/lstm_cell_20/BiasAddBiasAdd*sequential_10/lstm_20/lstm_cell_20/add:z:0Asequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2,
*sequential_10/lstm_20/lstm_cell_20/BiasAddк
2sequential_10/lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_10/lstm_20/lstm_cell_20/split/split_dim╧
(sequential_10/lstm_20/lstm_cell_20/splitSplit;sequential_10/lstm_20/lstm_cell_20/split/split_dim:output:03sequential_10/lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2*
(sequential_10/lstm_20/lstm_cell_20/split╔
*sequential_10/lstm_20/lstm_cell_20/SigmoidSigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2,
*sequential_10/lstm_20/lstm_cell_20/Sigmoid═
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_1Sigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2.
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_1ф
&sequential_10/lstm_20/lstm_cell_20/mulMul0sequential_10/lstm_20/lstm_cell_20/Sigmoid_1:y:0&sequential_10/lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:         Ю2(
&sequential_10/lstm_20/lstm_cell_20/mul└
'sequential_10/lstm_20/lstm_cell_20/ReluRelu1sequential_10/lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2)
'sequential_10/lstm_20/lstm_cell_20/Reluї
(sequential_10/lstm_20/lstm_cell_20/mul_1Mul.sequential_10/lstm_20/lstm_cell_20/Sigmoid:y:05sequential_10/lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2*
(sequential_10/lstm_20/lstm_cell_20/mul_1ъ
(sequential_10/lstm_20/lstm_cell_20/add_1AddV2*sequential_10/lstm_20/lstm_cell_20/mul:z:0,sequential_10/lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2*
(sequential_10/lstm_20/lstm_cell_20/add_1═
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_2Sigmoid1sequential_10/lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2.
,sequential_10/lstm_20/lstm_cell_20/Sigmoid_2┐
)sequential_10/lstm_20/lstm_cell_20/Relu_1Relu,sequential_10/lstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2+
)sequential_10/lstm_20/lstm_cell_20/Relu_1∙
(sequential_10/lstm_20/lstm_cell_20/mul_2Mul0sequential_10/lstm_20/lstm_cell_20/Sigmoid_2:y:07sequential_10/lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2*
(sequential_10/lstm_20/lstm_cell_20/mul_2╗
3sequential_10/lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   25
3sequential_10/lstm_20/TensorArrayV2_1/element_shapeР
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
sequential_10/lstm_20/timeл
.sequential_10/lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_10/lstm_20/while/maximum_iterationsЦ
(sequential_10/lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_10/lstm_20/while/loop_counter▌
sequential_10/lstm_20/whileWhile1sequential_10/lstm_20/while/loop_counter:output:07sequential_10/lstm_20/while/maximum_iterations:output:0#sequential_10/lstm_20/time:output:0.sequential_10/lstm_20/TensorArrayV2_1:handle:0$sequential_10/lstm_20/zeros:output:0&sequential_10/lstm_20/zeros_1:output:0.sequential_10/lstm_20/strided_slice_1:output:0Msequential_10/lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_10_lstm_20_lstm_cell_20_matmul_readvariableop_resourceCsequential_10_lstm_20_lstm_cell_20_matmul_1_readvariableop_resourceBsequential_10_lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_10_lstm_20_while_body_39100361*5
cond-R+
)sequential_10_lstm_20_while_cond_39100360*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
sequential_10/lstm_20/whileс
Fsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2H
Fsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shape┴
8sequential_10/lstm_20/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_10/lstm_20/while:output:3Osequential_10/lstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ю*
element_dtype02:
8sequential_10/lstm_20/TensorArrayV2Stack/TensorListStackн
+sequential_10/lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_10/lstm_20/strided_slice_3/stackи
-sequential_10/lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_10/lstm_20/strided_slice_3/stack_1и
-sequential_10/lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_20/strided_slice_3/stack_2Я
%sequential_10/lstm_20/strided_slice_3StridedSliceAsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack:tensor:04sequential_10/lstm_20/strided_slice_3/stack:output:06sequential_10/lstm_20/strided_slice_3/stack_1:output:06sequential_10/lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ю*
shrink_axis_mask2'
%sequential_10/lstm_20/strided_slice_3е
&sequential_10/lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_10/lstm_20/transpose_1/perm■
!sequential_10/lstm_20/transpose_1	TransposeAsequential_10/lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_10/lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ю2#
!sequential_10/lstm_20/transpose_1Т
sequential_10/lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_10/lstm_20/runtime░
!sequential_10/dropout_20/IdentityIdentity%sequential_10/lstm_20/transpose_1:y:0*
T0*,
_output_shapes
:         Ю2#
!sequential_10/dropout_20/IdentityФ
sequential_10/lstm_21/ShapeShape*sequential_10/dropout_20/Identity:output:0*
T0*
_output_shapes
:2
sequential_10/lstm_21/Shapeа
)sequential_10/lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_10/lstm_21/strided_slice/stackд
+sequential_10/lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_21/strided_slice/stack_1д
+sequential_10/lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_10/lstm_21/strided_slice/stack_2ц
#sequential_10/lstm_21/strided_sliceStridedSlice$sequential_10/lstm_21/Shape:output:02sequential_10/lstm_21/strided_slice/stack:output:04sequential_10/lstm_21/strided_slice/stack_1:output:04sequential_10/lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_10/lstm_21/strided_sliceИ
!sequential_10/lstm_21/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2#
!sequential_10/lstm_21/zeros/mul/y─
sequential_10/lstm_21/zeros/mulMul,sequential_10/lstm_21/strided_slice:output:0*sequential_10/lstm_21/zeros/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_21/zeros/mulЛ
"sequential_10/lstm_21/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2$
"sequential_10/lstm_21/zeros/Less/y┐
 sequential_10/lstm_21/zeros/LessLess#sequential_10/lstm_21/zeros/mul:z:0+sequential_10/lstm_21/zeros/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_10/lstm_21/zeros/LessО
$sequential_10/lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2&
$sequential_10/lstm_21/zeros/packed/1█
"sequential_10/lstm_21/zeros/packedPack,sequential_10/lstm_21/strided_slice:output:0-sequential_10/lstm_21/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_10/lstm_21/zeros/packedЛ
!sequential_10/lstm_21/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_10/lstm_21/zeros/Const═
sequential_10/lstm_21/zerosFill+sequential_10/lstm_21/zeros/packed:output:0*sequential_10/lstm_21/zeros/Const:output:0*
T0*'
_output_shapes
:         `2
sequential_10/lstm_21/zerosМ
#sequential_10/lstm_21/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2%
#sequential_10/lstm_21/zeros_1/mul/y╩
!sequential_10/lstm_21/zeros_1/mulMul,sequential_10/lstm_21/strided_slice:output:0,sequential_10/lstm_21/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_21/zeros_1/mulП
$sequential_10/lstm_21/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2&
$sequential_10/lstm_21/zeros_1/Less/y╟
"sequential_10/lstm_21/zeros_1/LessLess%sequential_10/lstm_21/zeros_1/mul:z:0-sequential_10/lstm_21/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2$
"sequential_10/lstm_21/zeros_1/LessТ
&sequential_10/lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2(
&sequential_10/lstm_21/zeros_1/packed/1с
$sequential_10/lstm_21/zeros_1/packedPack,sequential_10/lstm_21/strided_slice:output:0/sequential_10/lstm_21/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_10/lstm_21/zeros_1/packedП
#sequential_10/lstm_21/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#sequential_10/lstm_21/zeros_1/Const╒
sequential_10/lstm_21/zeros_1Fill-sequential_10/lstm_21/zeros_1/packed:output:0,sequential_10/lstm_21/zeros_1/Const:output:0*
T0*'
_output_shapes
:         `2
sequential_10/lstm_21/zeros_1б
$sequential_10/lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_10/lstm_21/transpose/permс
sequential_10/lstm_21/transpose	Transpose*sequential_10/dropout_20/Identity:output:0-sequential_10/lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:         Ю2!
sequential_10/lstm_21/transposeС
sequential_10/lstm_21/Shape_1Shape#sequential_10/lstm_21/transpose:y:0*
T0*
_output_shapes
:2
sequential_10/lstm_21/Shape_1д
+sequential_10/lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_21/strided_slice_1/stackи
-sequential_10/lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_1/stack_1и
-sequential_10/lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_1/stack_2Є
%sequential_10/lstm_21/strided_slice_1StridedSlice&sequential_10/lstm_21/Shape_1:output:04sequential_10/lstm_21/strided_slice_1/stack:output:06sequential_10/lstm_21/strided_slice_1/stack_1:output:06sequential_10/lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_1▒
1sequential_10/lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         23
1sequential_10/lstm_21/TensorArrayV2/element_shapeК
#sequential_10/lstm_21/TensorArrayV2TensorListReserve:sequential_10/lstm_21/TensorArrayV2/element_shape:output:0.sequential_10/lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_10/lstm_21/TensorArrayV2ы
Ksequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2M
Ksequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape╨
=sequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_10/lstm_21/transpose:y:0Tsequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=sequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensorд
+sequential_10/lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_10/lstm_21/strided_slice_2/stackи
-sequential_10/lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_2/stack_1и
-sequential_10/lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_2/stack_2Б
%sequential_10/lstm_21/strided_slice_2StridedSlice#sequential_10/lstm_21/transpose:y:04sequential_10/lstm_21/strided_slice_2/stack:output:06sequential_10/lstm_21/strided_slice_2/stack_1:output:06sequential_10/lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ю*
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_2°
8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOpAsequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02:
8sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOpЕ
)sequential_10/lstm_21/lstm_cell_21/MatMulMatMul.sequential_10/lstm_21/strided_slice_2:output:0@sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2+
)sequential_10/lstm_21/lstm_cell_21/MatMul¤
:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOpCsequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02<
:sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpБ
+sequential_10/lstm_21/lstm_cell_21/MatMul_1MatMul$sequential_10/lstm_21/zeros:output:0Bsequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2-
+sequential_10/lstm_21/lstm_cell_21/MatMul_1°
&sequential_10/lstm_21/lstm_cell_21/addAddV23sequential_10/lstm_21/lstm_cell_21/MatMul:product:05sequential_10/lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2(
&sequential_10/lstm_21/lstm_cell_21/addЎ
9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOpBsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02;
9sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpЕ
*sequential_10/lstm_21/lstm_cell_21/BiasAddBiasAdd*sequential_10/lstm_21/lstm_cell_21/add:z:0Asequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2,
*sequential_10/lstm_21/lstm_cell_21/BiasAddк
2sequential_10/lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :24
2sequential_10/lstm_21/lstm_cell_21/split/split_dim╦
(sequential_10/lstm_21/lstm_cell_21/splitSplit;sequential_10/lstm_21/lstm_cell_21/split/split_dim:output:03sequential_10/lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2*
(sequential_10/lstm_21/lstm_cell_21/split╚
*sequential_10/lstm_21/lstm_cell_21/SigmoidSigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2,
*sequential_10/lstm_21/lstm_cell_21/Sigmoid╠
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_1Sigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2.
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_1у
&sequential_10/lstm_21/lstm_cell_21/mulMul0sequential_10/lstm_21/lstm_cell_21/Sigmoid_1:y:0&sequential_10/lstm_21/zeros_1:output:0*
T0*'
_output_shapes
:         `2(
&sequential_10/lstm_21/lstm_cell_21/mul┐
'sequential_10/lstm_21/lstm_cell_21/ReluRelu1sequential_10/lstm_21/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2)
'sequential_10/lstm_21/lstm_cell_21/ReluЇ
(sequential_10/lstm_21/lstm_cell_21/mul_1Mul.sequential_10/lstm_21/lstm_cell_21/Sigmoid:y:05sequential_10/lstm_21/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2*
(sequential_10/lstm_21/lstm_cell_21/mul_1щ
(sequential_10/lstm_21/lstm_cell_21/add_1AddV2*sequential_10/lstm_21/lstm_cell_21/mul:z:0,sequential_10/lstm_21/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2*
(sequential_10/lstm_21/lstm_cell_21/add_1╠
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_2Sigmoid1sequential_10/lstm_21/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2.
,sequential_10/lstm_21/lstm_cell_21/Sigmoid_2╛
)sequential_10/lstm_21/lstm_cell_21/Relu_1Relu,sequential_10/lstm_21/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2+
)sequential_10/lstm_21/lstm_cell_21/Relu_1°
(sequential_10/lstm_21/lstm_cell_21/mul_2Mul0sequential_10/lstm_21/lstm_cell_21/Sigmoid_2:y:07sequential_10/lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2*
(sequential_10/lstm_21/lstm_cell_21/mul_2╗
3sequential_10/lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   25
3sequential_10/lstm_21/TensorArrayV2_1/element_shapeР
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
sequential_10/lstm_21/timeл
.sequential_10/lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         20
.sequential_10/lstm_21/while/maximum_iterationsЦ
(sequential_10/lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential_10/lstm_21/while/loop_counter┘
sequential_10/lstm_21/whileWhile1sequential_10/lstm_21/while/loop_counter:output:07sequential_10/lstm_21/while/maximum_iterations:output:0#sequential_10/lstm_21/time:output:0.sequential_10/lstm_21/TensorArrayV2_1:handle:0$sequential_10/lstm_21/zeros:output:0&sequential_10/lstm_21/zeros_1:output:0.sequential_10/lstm_21/strided_slice_1:output:0Msequential_10/lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:0Asequential_10_lstm_21_lstm_cell_21_matmul_readvariableop_resourceCsequential_10_lstm_21_lstm_cell_21_matmul_1_readvariableop_resourceBsequential_10_lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *5
body-R+
)sequential_10_lstm_21_while_body_39100509*5
cond-R+
)sequential_10_lstm_21_while_cond_39100508*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
sequential_10/lstm_21/whileс
Fsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2H
Fsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shape└
8sequential_10/lstm_21/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_10/lstm_21/while:output:3Osequential_10/lstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         `*
element_dtype02:
8sequential_10/lstm_21/TensorArrayV2Stack/TensorListStackн
+sequential_10/lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2-
+sequential_10/lstm_21/strided_slice_3/stackи
-sequential_10/lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential_10/lstm_21/strided_slice_3/stack_1и
-sequential_10/lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-sequential_10/lstm_21/strided_slice_3/stack_2Ю
%sequential_10/lstm_21/strided_slice_3StridedSliceAsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:04sequential_10/lstm_21/strided_slice_3/stack:output:06sequential_10/lstm_21/strided_slice_3/stack_1:output:06sequential_10/lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         `*
shrink_axis_mask2'
%sequential_10/lstm_21/strided_slice_3е
&sequential_10/lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&sequential_10/lstm_21/transpose_1/perm¤
!sequential_10/lstm_21/transpose_1	TransposeAsequential_10/lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_10/lstm_21/transpose_1/perm:output:0*
T0*+
_output_shapes
:         `2#
!sequential_10/lstm_21/transpose_1Т
sequential_10/lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_10/lstm_21/runtimeп
!sequential_10/dropout_21/IdentityIdentity%sequential_10/lstm_21/transpose_1:y:0*
T0*+
_output_shapes
:         `2#
!sequential_10/dropout_21/Identity█
/sequential_10/dense_10/Tensordot/ReadVariableOpReadVariableOp8sequential_10_dense_10_tensordot_readvariableop_resource*
_output_shapes

:`*
dtype021
/sequential_10/dense_10/Tensordot/ReadVariableOpШ
%sequential_10/dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2'
%sequential_10/dense_10/Tensordot/axesЯ
%sequential_10/dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2'
%sequential_10/dense_10/Tensordot/freeк
&sequential_10/dense_10/Tensordot/ShapeShape*sequential_10/dropout_21/Identity:output:0*
T0*
_output_shapes
:2(
&sequential_10/dense_10/Tensordot/Shapeв
.sequential_10/dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_10/Tensordot/GatherV2/axis─
)sequential_10/dense_10/Tensordot/GatherV2GatherV2/sequential_10/dense_10/Tensordot/Shape:output:0.sequential_10/dense_10/Tensordot/free:output:07sequential_10/dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_10/dense_10/Tensordot/GatherV2ж
0sequential_10/dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 22
0sequential_10/dense_10/Tensordot/GatherV2_1/axis╩
+sequential_10/dense_10/Tensordot/GatherV2_1GatherV2/sequential_10/dense_10/Tensordot/Shape:output:0.sequential_10/dense_10/Tensordot/axes:output:09sequential_10/dense_10/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2-
+sequential_10/dense_10/Tensordot/GatherV2_1Ъ
&sequential_10/dense_10/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_10/dense_10/Tensordot/Const▄
%sequential_10/dense_10/Tensordot/ProdProd2sequential_10/dense_10/Tensordot/GatherV2:output:0/sequential_10/dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2'
%sequential_10/dense_10/Tensordot/ProdЮ
(sequential_10/dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_10/dense_10/Tensordot/Const_1ф
'sequential_10/dense_10/Tensordot/Prod_1Prod4sequential_10/dense_10/Tensordot/GatherV2_1:output:01sequential_10/dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2)
'sequential_10/dense_10/Tensordot/Prod_1Ю
,sequential_10/dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_10/dense_10/Tensordot/concat/axisг
'sequential_10/dense_10/Tensordot/concatConcatV2.sequential_10/dense_10/Tensordot/free:output:0.sequential_10/dense_10/Tensordot/axes:output:05sequential_10/dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_10/dense_10/Tensordot/concatш
&sequential_10/dense_10/Tensordot/stackPack.sequential_10/dense_10/Tensordot/Prod:output:00sequential_10/dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2(
&sequential_10/dense_10/Tensordot/stack∙
*sequential_10/dense_10/Tensordot/transpose	Transpose*sequential_10/dropout_21/Identity:output:00sequential_10/dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:         `2,
*sequential_10/dense_10/Tensordot/transpose√
(sequential_10/dense_10/Tensordot/ReshapeReshape.sequential_10/dense_10/Tensordot/transpose:y:0/sequential_10/dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2*
(sequential_10/dense_10/Tensordot/Reshape·
'sequential_10/dense_10/Tensordot/MatMulMatMul1sequential_10/dense_10/Tensordot/Reshape:output:07sequential_10/dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2)
'sequential_10/dense_10/Tensordot/MatMulЮ
(sequential_10/dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(sequential_10/dense_10/Tensordot/Const_2в
.sequential_10/dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_10/dense_10/Tensordot/concat_1/axis░
)sequential_10/dense_10/Tensordot/concat_1ConcatV22sequential_10/dense_10/Tensordot/GatherV2:output:01sequential_10/dense_10/Tensordot/Const_2:output:07sequential_10/dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2+
)sequential_10/dense_10/Tensordot/concat_1ь
 sequential_10/dense_10/TensordotReshape1sequential_10/dense_10/Tensordot/MatMul:product:02sequential_10/dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2"
 sequential_10/dense_10/Tensordot╤
-sequential_10/dense_10/BiasAdd/ReadVariableOpReadVariableOp6sequential_10_dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_10/dense_10/BiasAdd/ReadVariableOpу
sequential_10/dense_10/BiasAddBiasAdd)sequential_10/dense_10/Tensordot:output:05sequential_10/dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2 
sequential_10/dense_10/BiasAddк
sequential_10/dense_10/SoftmaxSoftmax'sequential_10/dense_10/BiasAdd:output:0*
T0*+
_output_shapes
:         2 
sequential_10/dense_10/SoftmaxЗ
IdentityIdentity(sequential_10/dense_10/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╘
NoOpNoOp.^sequential_10/dense_10/BiasAdd/ReadVariableOp0^sequential_10/dense_10/Tensordot/ReadVariableOp:^sequential_10/lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp9^sequential_10/lstm_20/lstm_cell_20/MatMul/ReadVariableOp;^sequential_10/lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^sequential_10/lstm_20/while:^sequential_10/lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp9^sequential_10/lstm_21/lstm_cell_21/MatMul/ReadVariableOp;^sequential_10/lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^sequential_10/lstm_21/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2^
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
:         ]
'
_user_specified_namelstm_20_input
р
║
*__inference_lstm_21_layer_call_fn_39104855
inputs_0
unknown:
ЮА
	unknown_0:	`А
	unknown_1:	А
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391016192
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  Ю
"
_user_specified_name
inputs/0
∙
Е
)sequential_10_lstm_20_while_cond_39100360H
Dsequential_10_lstm_20_while_sequential_10_lstm_20_while_loop_counterN
Jsequential_10_lstm_20_while_sequential_10_lstm_20_while_maximum_iterations+
'sequential_10_lstm_20_while_placeholder-
)sequential_10_lstm_20_while_placeholder_1-
)sequential_10_lstm_20_while_placeholder_2-
)sequential_10_lstm_20_while_placeholder_3J
Fsequential_10_lstm_20_while_less_sequential_10_lstm_20_strided_slice_1b
^sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_39100360___redundant_placeholder0b
^sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_39100360___redundant_placeholder1b
^sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_39100360___redundant_placeholder2b
^sequential_10_lstm_20_while_sequential_10_lstm_20_while_cond_39100360___redundant_placeholder3(
$sequential_10_lstm_20_while_identity
▐
 sequential_10/lstm_20/while/LessLess'sequential_10_lstm_20_while_placeholderFsequential_10_lstm_20_while_less_sequential_10_lstm_20_strided_slice_1*
T0*
_output_shapes
: 2"
 sequential_10/lstm_20/while/LessЯ
$sequential_10/lstm_20/while/IdentityIdentity$sequential_10/lstm_20/while/Less:z:0*
T0
*
_output_shapes
: 2&
$sequential_10/lstm_20/while/Identity"U
$sequential_10_lstm_20_while_identity-sequential_10/lstm_20/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
╫
g
H__inference_dropout_20_layer_call_and_return_conditional_losses_39102501

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
:         Ю2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ю*
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
:         Ю2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ю2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ю2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ю2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ю:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
╙J
╘

lstm_21_while_body_39103393,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3+
'lstm_21_while_lstm_21_strided_slice_1_0g
clstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮАP
=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АK
<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	А
lstm_21_while_identity
lstm_21_while_identity_1
lstm_21_while_identity_2
lstm_21_while_identity_3
lstm_21_while_identity_4
lstm_21_while_identity_5)
%lstm_21_while_lstm_21_strided_slice_1e
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorM
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:
ЮАN
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:	`АI
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	АИв1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpв0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpв2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp╙
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2A
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0lstm_21_while_placeholderHlstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype023
1lstm_21/while/TensorArrayV2Read/TensorListGetItemт
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype022
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpў
!lstm_21/while/lstm_cell_21/MatMulMatMul8lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_21/while/lstm_cell_21/MatMulч
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype024
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpр
#lstm_21/while/lstm_cell_21/MatMul_1MatMullstm_21_while_placeholder_2:lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2%
#lstm_21/while/lstm_cell_21/MatMul_1╪
lstm_21/while/lstm_cell_21/addAddV2+lstm_21/while/lstm_cell_21/MatMul:product:0-lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2 
lstm_21/while/lstm_cell_21/addр
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype023
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpх
"lstm_21/while/lstm_cell_21/BiasAddBiasAdd"lstm_21/while/lstm_cell_21/add:z:09lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2$
"lstm_21/while/lstm_cell_21/BiasAddЪ
*lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_21/while/lstm_cell_21/split/split_dimл
 lstm_21/while/lstm_cell_21/splitSplit3lstm_21/while/lstm_cell_21/split/split_dim:output:0+lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2"
 lstm_21/while/lstm_cell_21/split░
"lstm_21/while/lstm_cell_21/SigmoidSigmoid)lstm_21/while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2$
"lstm_21/while/lstm_cell_21/Sigmoid┤
$lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid)lstm_21/while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2&
$lstm_21/while/lstm_cell_21/Sigmoid_1└
lstm_21/while/lstm_cell_21/mulMul(lstm_21/while/lstm_cell_21/Sigmoid_1:y:0lstm_21_while_placeholder_3*
T0*'
_output_shapes
:         `2 
lstm_21/while/lstm_cell_21/mulз
lstm_21/while/lstm_cell_21/ReluRelu)lstm_21/while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2!
lstm_21/while/lstm_cell_21/Relu╘
 lstm_21/while/lstm_cell_21/mul_1Mul&lstm_21/while/lstm_cell_21/Sigmoid:y:0-lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2"
 lstm_21/while/lstm_cell_21/mul_1╔
 lstm_21/while/lstm_cell_21/add_1AddV2"lstm_21/while/lstm_cell_21/mul:z:0$lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2"
 lstm_21/while/lstm_cell_21/add_1┤
$lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid)lstm_21/while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2&
$lstm_21/while/lstm_cell_21/Sigmoid_2ж
!lstm_21/while/lstm_cell_21/Relu_1Relu$lstm_21/while/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2#
!lstm_21/while/lstm_cell_21/Relu_1╪
 lstm_21/while/lstm_cell_21/mul_2Mul(lstm_21/while/lstm_cell_21/Sigmoid_2:y:0/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2"
 lstm_21/while/lstm_cell_21/mul_2И
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
lstm_21/while/add/yЙ
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
lstm_21/while/add_1/yЮ
lstm_21/while/add_1AddV2(lstm_21_while_lstm_21_while_loop_counterlstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/add_1Л
lstm_21/while/IdentityIdentitylstm_21/while/add_1:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identityж
lstm_21/while/Identity_1Identity.lstm_21_while_lstm_21_while_maximum_iterations^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_1Н
lstm_21/while/Identity_2Identitylstm_21/while/add:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_2║
lstm_21/while/Identity_3IdentityBlstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_3н
lstm_21/while/Identity_4Identity$lstm_21/while/lstm_cell_21/mul_2:z:0^lstm_21/while/NoOp*
T0*'
_output_shapes
:         `2
lstm_21/while/Identity_4н
lstm_21/while/Identity_5Identity$lstm_21/while/lstm_cell_21/add_1:z:0^lstm_21/while/NoOp*
T0*'
_output_shapes
:         `2
lstm_21/while/Identity_5Ж
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
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"╚
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2f
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
м\
а
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104531
inputs_0?
+lstm_cell_21_matmul_readvariableop_resource:
ЮА@
-lstm_cell_21_matmul_1_readvariableop_resource:	`А;
,lstm_cell_21_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_21/BiasAdd/ReadVariableOpв"lstm_cell_21/MatMul/ReadVariableOpв$lstm_cell_21/MatMul_1/ReadVariableOpвwhileF
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
value	B :`2
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
:         `2
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
value	B :`2
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
:         `2	
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
!:                  Ю2
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
valueB"    Ю   27
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
:         Ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOpн
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul╗
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOpй
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul_1а
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/add┤
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOpн
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dimє
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
lstm_cell_21/splitЖ
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/SigmoidК
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_1Л
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul}
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
lstm_cell_21/ReluЬ
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_1С
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/add_1К
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_2|
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/Relu_1а
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104447*
condR
while_cond_39104446*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
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
:         `*
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
 :                  `2
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
 :                  `2

Identity╚
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Ю: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  Ю
"
_user_specified_name
inputs/0
хJ
╘

lstm_20_while_body_39102911,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3+
'lstm_20_while_lstm_20_strided_slice_1_0g
clstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	]°Q
=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°K
<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	°
lstm_20_while_identity
lstm_20_while_identity_1
lstm_20_while_identity_2
lstm_20_while_identity_3
lstm_20_while_identity_4
lstm_20_while_identity_5)
%lstm_20_while_lstm_20_strided_slice_1e
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorL
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	]°O
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°I
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpв0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpв2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp╙
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0lstm_20_while_placeholderHlstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_20/while/TensorArrayV2Read/TensorListGetItemс
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype022
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpў
!lstm_20/while/lstm_cell_20/MatMulMatMul8lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2#
!lstm_20/while/lstm_cell_20/MatMulш
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype024
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpр
#lstm_20/while/lstm_cell_20/MatMul_1MatMullstm_20_while_placeholder_2:lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2%
#lstm_20/while/lstm_cell_20/MatMul_1╪
lstm_20/while/lstm_cell_20/addAddV2+lstm_20/while/lstm_cell_20/MatMul:product:0-lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2 
lstm_20/while/lstm_cell_20/addр
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype023
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpх
"lstm_20/while/lstm_cell_20/BiasAddBiasAdd"lstm_20/while/lstm_cell_20/add:z:09lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2$
"lstm_20/while/lstm_cell_20/BiasAddЪ
*lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_20/while/lstm_cell_20/split/split_dimп
 lstm_20/while/lstm_cell_20/splitSplit3lstm_20/while/lstm_cell_20/split/split_dim:output:0+lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2"
 lstm_20/while/lstm_cell_20/split▒
"lstm_20/while/lstm_cell_20/SigmoidSigmoid)lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2$
"lstm_20/while/lstm_cell_20/Sigmoid╡
$lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid)lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2&
$lstm_20/while/lstm_cell_20/Sigmoid_1┴
lstm_20/while/lstm_cell_20/mulMul(lstm_20/while/lstm_cell_20/Sigmoid_1:y:0lstm_20_while_placeholder_3*
T0*(
_output_shapes
:         Ю2 
lstm_20/while/lstm_cell_20/mulи
lstm_20/while/lstm_cell_20/ReluRelu)lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2!
lstm_20/while/lstm_cell_20/Relu╒
 lstm_20/while/lstm_cell_20/mul_1Mul&lstm_20/while/lstm_cell_20/Sigmoid:y:0-lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2"
 lstm_20/while/lstm_cell_20/mul_1╩
 lstm_20/while/lstm_cell_20/add_1AddV2"lstm_20/while/lstm_cell_20/mul:z:0$lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2"
 lstm_20/while/lstm_cell_20/add_1╡
$lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid)lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2&
$lstm_20/while/lstm_cell_20/Sigmoid_2з
!lstm_20/while/lstm_cell_20/Relu_1Relu$lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2#
!lstm_20/while/lstm_cell_20/Relu_1┘
 lstm_20/while/lstm_cell_20/mul_2Mul(lstm_20/while/lstm_cell_20/Sigmoid_2:y:0/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2"
 lstm_20/while/lstm_cell_20/mul_2И
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
lstm_20/while/add/yЙ
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
lstm_20/while/add_1/yЮ
lstm_20/while/add_1AddV2(lstm_20_while_lstm_20_while_loop_counterlstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/add_1Л
lstm_20/while/IdentityIdentitylstm_20/while/add_1:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identityж
lstm_20/while/Identity_1Identity.lstm_20_while_lstm_20_while_maximum_iterations^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_1Н
lstm_20/while/Identity_2Identitylstm_20/while/add:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_2║
lstm_20/while/Identity_3IdentityBlstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_3о
lstm_20/while/Identity_4Identity$lstm_20/while/lstm_cell_20/mul_2:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:         Ю2
lstm_20/while/Identity_4о
lstm_20/while/Identity_5Identity$lstm_20/while/lstm_cell_20/add_1:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:         Ю2
lstm_20/while/Identity_5Ж
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
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"╚
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2f
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
р
║
*__inference_lstm_20_layer_call_fn_39104180
inputs_0
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391009892
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  Ю2

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
┤∙
К
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103171

inputsF
3lstm_20_lstm_cell_20_matmul_readvariableop_resource:	]°I
5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°C
4lstm_20_lstm_cell_20_biasadd_readvariableop_resource:	°G
3lstm_21_lstm_cell_21_matmul_readvariableop_resource:
ЮАH
5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource:	`АC
4lstm_21_lstm_cell_21_biasadd_readvariableop_resource:	А<
*dense_10_tensordot_readvariableop_resource:`6
(dense_10_biasadd_readvariableop_resource:
identityИвdense_10/BiasAdd/ReadVariableOpв!dense_10/Tensordot/ReadVariableOpв+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpв*lstm_20/lstm_cell_20/MatMul/ReadVariableOpв,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpвlstm_20/whileв+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpв*lstm_21/lstm_cell_21/MatMul/ReadVariableOpв,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpвlstm_21/whileT
lstm_20/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_20/ShapeД
lstm_20/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice/stackИ
lstm_20/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_1И
lstm_20/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_20/strided_slice/stack_2Т
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
B :Ю2
lstm_20/zeros/mul/yМ
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
B :ш2
lstm_20/zeros/Less/yЗ
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
B :Ю2
lstm_20/zeros/packed/1г
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
lstm_20/zeros/ConstЦ
lstm_20/zerosFilllstm_20/zeros/packed:output:0lstm_20/zeros/Const:output:0*
T0*(
_output_shapes
:         Ю2
lstm_20/zerosq
lstm_20/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
lstm_20/zeros_1/mul/yТ
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
B :ш2
lstm_20/zeros_1/Less/yП
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
B :Ю2
lstm_20/zeros_1/packed/1й
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
lstm_20/zeros_1/ConstЮ
lstm_20/zeros_1Filllstm_20/zeros_1/packed:output:0lstm_20/zeros_1/Const:output:0*
T0*(
_output_shapes
:         Ю2
lstm_20/zeros_1Е
lstm_20/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose/permТ
lstm_20/transpose	Transposeinputslstm_20/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_20/transposeg
lstm_20/Shape_1Shapelstm_20/transpose:y:0*
T0*
_output_shapes
:2
lstm_20/Shape_1И
lstm_20/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_1/stackМ
lstm_20/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_1М
lstm_20/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_1/stack_2Ю
lstm_20/strided_slice_1StridedSlicelstm_20/Shape_1:output:0&lstm_20/strided_slice_1/stack:output:0(lstm_20/strided_slice_1/stack_1:output:0(lstm_20/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_20/strided_slice_1Х
#lstm_20/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_20/TensorArrayV2/element_shape╥
lstm_20/TensorArrayV2TensorListReserve,lstm_20/TensorArrayV2/element_shape:output:0 lstm_20/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_20/TensorArrayV2╧
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_20/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_20/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_20/transpose:y:0Flstm_20/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_20/TensorArrayUnstack/TensorListFromTensorИ
lstm_20/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_20/strided_slice_2/stackМ
lstm_20/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_1М
lstm_20/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_2/stack_2м
lstm_20/strided_slice_2StridedSlicelstm_20/transpose:y:0&lstm_20/strided_slice_2/stack:output:0(lstm_20/strided_slice_2/stack_1:output:0(lstm_20/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_20/strided_slice_2═
*lstm_20/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3lstm_20_lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02,
*lstm_20/lstm_cell_20/MatMul/ReadVariableOp═
lstm_20/lstm_cell_20/MatMulMatMul lstm_20/strided_slice_2:output:02lstm_20/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_20/lstm_cell_20/MatMul╘
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02.
,lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp╔
lstm_20/lstm_cell_20/MatMul_1MatMullstm_20/zeros:output:04lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_20/lstm_cell_20/MatMul_1└
lstm_20/lstm_cell_20/addAddV2%lstm_20/lstm_cell_20/MatMul:product:0'lstm_20/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
lstm_20/lstm_cell_20/add╠
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02-
+lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp═
lstm_20/lstm_cell_20/BiasAddBiasAddlstm_20/lstm_cell_20/add:z:03lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_20/lstm_cell_20/BiasAddО
$lstm_20/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_20/lstm_cell_20/split/split_dimЧ
lstm_20/lstm_cell_20/splitSplit-lstm_20/lstm_cell_20/split/split_dim:output:0%lstm_20/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
lstm_20/lstm_cell_20/splitЯ
lstm_20/lstm_cell_20/SigmoidSigmoid#lstm_20/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/Sigmoidг
lstm_20/lstm_cell_20/Sigmoid_1Sigmoid#lstm_20/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2 
lstm_20/lstm_cell_20/Sigmoid_1м
lstm_20/lstm_cell_20/mulMul"lstm_20/lstm_cell_20/Sigmoid_1:y:0lstm_20/zeros_1:output:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/mulЦ
lstm_20/lstm_cell_20/ReluRelu#lstm_20/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/Relu╜
lstm_20/lstm_cell_20/mul_1Mul lstm_20/lstm_cell_20/Sigmoid:y:0'lstm_20/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/mul_1▓
lstm_20/lstm_cell_20/add_1AddV2lstm_20/lstm_cell_20/mul:z:0lstm_20/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/add_1г
lstm_20/lstm_cell_20/Sigmoid_2Sigmoid#lstm_20/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2 
lstm_20/lstm_cell_20/Sigmoid_2Х
lstm_20/lstm_cell_20/Relu_1Relulstm_20/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/Relu_1┴
lstm_20/lstm_cell_20/mul_2Mul"lstm_20/lstm_cell_20/Sigmoid_2:y:0)lstm_20/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_20/lstm_cell_20/mul_2Я
%lstm_20/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2'
%lstm_20/TensorArrayV2_1/element_shape╪
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
lstm_20/timeП
 lstm_20/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_20/while/maximum_iterationsz
lstm_20/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_20/while/loop_counterЛ
lstm_20/whileWhile#lstm_20/while/loop_counter:output:0)lstm_20/while/maximum_iterations:output:0lstm_20/time:output:0 lstm_20/TensorArrayV2_1:handle:0lstm_20/zeros:output:0lstm_20/zeros_1:output:0 lstm_20/strided_slice_1:output:0?lstm_20/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_20_lstm_cell_20_matmul_readvariableop_resource5lstm_20_lstm_cell_20_matmul_1_readvariableop_resource4lstm_20_lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_20_while_body_39102911*'
condR
lstm_20_while_cond_39102910*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
lstm_20/while┼
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2:
8lstm_20/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_20/TensorArrayV2Stack/TensorListStackTensorListStacklstm_20/while:output:3Alstm_20/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ю*
element_dtype02,
*lstm_20/TensorArrayV2Stack/TensorListStackС
lstm_20/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_20/strided_slice_3/stackМ
lstm_20/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_20/strided_slice_3/stack_1М
lstm_20/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_20/strided_slice_3/stack_2╦
lstm_20/strided_slice_3StridedSlice3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_20/strided_slice_3/stack:output:0(lstm_20/strided_slice_3/stack_1:output:0(lstm_20/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ю*
shrink_axis_mask2
lstm_20/strided_slice_3Й
lstm_20/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_20/transpose_1/perm╞
lstm_20/transpose_1	Transpose3lstm_20/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_20/transpose_1/perm:output:0*
T0*,
_output_shapes
:         Ю2
lstm_20/transpose_1v
lstm_20/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_20/runtimeЖ
dropout_20/IdentityIdentitylstm_20/transpose_1:y:0*
T0*,
_output_shapes
:         Ю2
dropout_20/Identityj
lstm_21/ShapeShapedropout_20/Identity:output:0*
T0*
_output_shapes
:2
lstm_21/ShapeД
lstm_21/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice/stackИ
lstm_21/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_1И
lstm_21/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_21/strided_slice/stack_2Т
lstm_21/strided_sliceStridedSlicelstm_21/Shape:output:0$lstm_21/strided_slice/stack:output:0&lstm_21/strided_slice/stack_1:output:0&lstm_21/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slicel
lstm_21/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
lstm_21/zeros/mul/yМ
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
B :ш2
lstm_21/zeros/Less/yЗ
lstm_21/zeros/LessLesslstm_21/zeros/mul:z:0lstm_21/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros/Lessr
lstm_21/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2
lstm_21/zeros/packed/1г
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
lstm_21/zeros/ConstХ
lstm_21/zerosFilllstm_21/zeros/packed:output:0lstm_21/zeros/Const:output:0*
T0*'
_output_shapes
:         `2
lstm_21/zerosp
lstm_21/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :`2
lstm_21/zeros_1/mul/yТ
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
B :ш2
lstm_21/zeros_1/Less/yП
lstm_21/zeros_1/LessLesslstm_21/zeros_1/mul:z:0lstm_21/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_21/zeros_1/Lessv
lstm_21/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :`2
lstm_21/zeros_1/packed/1й
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
lstm_21/zeros_1/ConstЭ
lstm_21/zeros_1Filllstm_21/zeros_1/packed:output:0lstm_21/zeros_1/Const:output:0*
T0*'
_output_shapes
:         `2
lstm_21/zeros_1Е
lstm_21/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose/permй
lstm_21/transpose	Transposedropout_20/Identity:output:0lstm_21/transpose/perm:output:0*
T0*,
_output_shapes
:         Ю2
lstm_21/transposeg
lstm_21/Shape_1Shapelstm_21/transpose:y:0*
T0*
_output_shapes
:2
lstm_21/Shape_1И
lstm_21/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_1/stackМ
lstm_21/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_1М
lstm_21/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_1/stack_2Ю
lstm_21/strided_slice_1StridedSlicelstm_21/Shape_1:output:0&lstm_21/strided_slice_1/stack:output:0(lstm_21/strided_slice_1/stack_1:output:0(lstm_21/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_21/strided_slice_1Х
#lstm_21/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_21/TensorArrayV2/element_shape╥
lstm_21/TensorArrayV2TensorListReserve,lstm_21/TensorArrayV2/element_shape:output:0 lstm_21/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_21/TensorArrayV2╧
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2?
=lstm_21/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_21/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_21/transpose:y:0Flstm_21/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_21/TensorArrayUnstack/TensorListFromTensorИ
lstm_21/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_21/strided_slice_2/stackМ
lstm_21/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_1М
lstm_21/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_2/stack_2н
lstm_21/strided_slice_2StridedSlicelstm_21/transpose:y:0&lstm_21/strided_slice_2/stack:output:0(lstm_21/strided_slice_2/stack_1:output:0(lstm_21/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         Ю*
shrink_axis_mask2
lstm_21/strided_slice_2╬
*lstm_21/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3lstm_21_lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02,
*lstm_21/lstm_cell_21/MatMul/ReadVariableOp═
lstm_21/lstm_cell_21/MatMulMatMul lstm_21/strided_slice_2:output:02lstm_21/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_21/lstm_cell_21/MatMul╙
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02.
,lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp╔
lstm_21/lstm_cell_21/MatMul_1MatMullstm_21/zeros:output:04lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_21/lstm_cell_21/MatMul_1└
lstm_21/lstm_cell_21/addAddV2%lstm_21/lstm_cell_21/MatMul:product:0'lstm_21/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_21/lstm_cell_21/add╠
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02-
+lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp═
lstm_21/lstm_cell_21/BiasAddBiasAddlstm_21/lstm_cell_21/add:z:03lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_21/lstm_cell_21/BiasAddО
$lstm_21/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_21/lstm_cell_21/split/split_dimУ
lstm_21/lstm_cell_21/splitSplit-lstm_21/lstm_cell_21/split/split_dim:output:0%lstm_21/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
lstm_21/lstm_cell_21/splitЮ
lstm_21/lstm_cell_21/SigmoidSigmoid#lstm_21/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/Sigmoidв
lstm_21/lstm_cell_21/Sigmoid_1Sigmoid#lstm_21/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2 
lstm_21/lstm_cell_21/Sigmoid_1л
lstm_21/lstm_cell_21/mulMul"lstm_21/lstm_cell_21/Sigmoid_1:y:0lstm_21/zeros_1:output:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/mulХ
lstm_21/lstm_cell_21/ReluRelu#lstm_21/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/Relu╝
lstm_21/lstm_cell_21/mul_1Mul lstm_21/lstm_cell_21/Sigmoid:y:0'lstm_21/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/mul_1▒
lstm_21/lstm_cell_21/add_1AddV2lstm_21/lstm_cell_21/mul:z:0lstm_21/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/add_1в
lstm_21/lstm_cell_21/Sigmoid_2Sigmoid#lstm_21/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2 
lstm_21/lstm_cell_21/Sigmoid_2Ф
lstm_21/lstm_cell_21/Relu_1Relulstm_21/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/Relu_1└
lstm_21/lstm_cell_21/mul_2Mul"lstm_21/lstm_cell_21/Sigmoid_2:y:0)lstm_21/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
lstm_21/lstm_cell_21/mul_2Я
%lstm_21/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2'
%lstm_21/TensorArrayV2_1/element_shape╪
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
lstm_21/timeП
 lstm_21/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_21/while/maximum_iterationsz
lstm_21/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_21/while/loop_counterЗ
lstm_21/whileWhile#lstm_21/while/loop_counter:output:0)lstm_21/while/maximum_iterations:output:0lstm_21/time:output:0 lstm_21/TensorArrayV2_1:handle:0lstm_21/zeros:output:0lstm_21/zeros_1:output:0 lstm_21/strided_slice_1:output:0?lstm_21/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_21_lstm_cell_21_matmul_readvariableop_resource5lstm_21_lstm_cell_21_matmul_1_readvariableop_resource4lstm_21_lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_21_while_body_39103059*'
condR
lstm_21_while_cond_39103058*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
lstm_21/while┼
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2:
8lstm_21/TensorArrayV2Stack/TensorListStack/element_shapeИ
*lstm_21/TensorArrayV2Stack/TensorListStackTensorListStacklstm_21/while:output:3Alstm_21/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         `*
element_dtype02,
*lstm_21/TensorArrayV2Stack/TensorListStackС
lstm_21/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_21/strided_slice_3/stackМ
lstm_21/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_21/strided_slice_3/stack_1М
lstm_21/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_21/strided_slice_3/stack_2╩
lstm_21/strided_slice_3StridedSlice3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_21/strided_slice_3/stack:output:0(lstm_21/strided_slice_3/stack_1:output:0(lstm_21/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         `*
shrink_axis_mask2
lstm_21/strided_slice_3Й
lstm_21/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_21/transpose_1/perm┼
lstm_21/transpose_1	Transpose3lstm_21/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_21/transpose_1/perm:output:0*
T0*+
_output_shapes
:         `2
lstm_21/transpose_1v
lstm_21/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_21/runtimeЕ
dropout_21/IdentityIdentitylstm_21/transpose_1:y:0*
T0*+
_output_shapes
:         `2
dropout_21/Identity▒
!dense_10/Tensordot/ReadVariableOpReadVariableOp*dense_10_tensordot_readvariableop_resource*
_output_shapes

:`*
dtype02#
!dense_10/Tensordot/ReadVariableOp|
dense_10/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/axesГ
dense_10/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_10/Tensordot/freeА
dense_10/Tensordot/ShapeShapedropout_21/Identity:output:0*
T0*
_output_shapes
:2
dense_10/Tensordot/ShapeЖ
 dense_10/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/GatherV2/axis■
dense_10/Tensordot/GatherV2GatherV2!dense_10/Tensordot/Shape:output:0 dense_10/Tensordot/free:output:0)dense_10/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_10/Tensordot/GatherV2К
"dense_10/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"dense_10/Tensordot/GatherV2_1/axisД
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
dense_10/Tensordot/Constд
dense_10/Tensordot/ProdProd$dense_10/Tensordot/GatherV2:output:0!dense_10/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/ProdВ
dense_10/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_10/Tensordot/Const_1м
dense_10/Tensordot/Prod_1Prod&dense_10/Tensordot/GatherV2_1:output:0#dense_10/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_10/Tensordot/Prod_1В
dense_10/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
dense_10/Tensordot/concat/axis▌
dense_10/Tensordot/concatConcatV2 dense_10/Tensordot/free:output:0 dense_10/Tensordot/axes:output:0'dense_10/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat░
dense_10/Tensordot/stackPack dense_10/Tensordot/Prod:output:0"dense_10/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/stack┴
dense_10/Tensordot/transpose	Transposedropout_21/Identity:output:0"dense_10/Tensordot/concat:output:0*
T0*+
_output_shapes
:         `2
dense_10/Tensordot/transpose├
dense_10/Tensordot/ReshapeReshape dense_10/Tensordot/transpose:y:0!dense_10/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_10/Tensordot/Reshape┬
dense_10/Tensordot/MatMulMatMul#dense_10/Tensordot/Reshape:output:0)dense_10/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_10/Tensordot/MatMulВ
dense_10/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_10/Tensordot/Const_2Ж
 dense_10/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2"
 dense_10/Tensordot/concat_1/axisъ
dense_10/Tensordot/concat_1ConcatV2$dense_10/Tensordot/GatherV2:output:0#dense_10/Tensordot/Const_2:output:0)dense_10/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_10/Tensordot/concat_1┤
dense_10/TensordotReshape#dense_10/Tensordot/MatMul:product:0$dense_10/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_10/Tensordotз
dense_10/BiasAdd/ReadVariableOpReadVariableOp(dense_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_10/BiasAdd/ReadVariableOpл
dense_10/BiasAddBiasAdddense_10/Tensordot:output:0'dense_10/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_10/BiasAddА
dense_10/SoftmaxSoftmaxdense_10/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_10/Softmaxy
IdentityIdentitydense_10/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp ^dense_10/BiasAdd/ReadVariableOp"^dense_10/Tensordot/ReadVariableOp,^lstm_20/lstm_cell_20/BiasAdd/ReadVariableOp+^lstm_20/lstm_cell_20/MatMul/ReadVariableOp-^lstm_20/lstm_cell_20/MatMul_1/ReadVariableOp^lstm_20/while,^lstm_21/lstm_cell_21/BiasAdd/ReadVariableOp+^lstm_21/lstm_cell_21/MatMul/ReadVariableOp-^lstm_21/lstm_cell_21/MatMul_1/ReadVariableOp^lstm_21/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
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
:         ]
 
_user_specified_nameinputs
Д\
Ю
E__inference_lstm_20_layer_call_and_return_conditional_losses_39102038

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	]°A
-lstm_cell_20_matmul_1_readvariableop_resource:
Ю°;
,lstm_cell_20_biasadd_readvariableop_resource:	°
identityИв#lstm_cell_20/BiasAdd/ReadVariableOpв"lstm_cell_20/MatMul/ReadVariableOpв$lstm_cell_20/MatMul_1/ReadVariableOpвwhileD
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
B :Ю2
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
B :Ю2
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
:         Ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
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
B :Ю2
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
:         Ю2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOpн
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul╝
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOpй
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul_1а
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/add┤
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOpн
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dimў
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
lstm_cell_20/splitЗ
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/SigmoidЛ
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_1М
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/ReluЭ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_1Т
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/add_1Л
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Relu_1б
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39101954*
condR
while_cond_39101953*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ю*
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
:         Ю*
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
:         Ю2
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
:         Ю2

Identity╚
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
∙
З
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39101326

inputs

states
states_12
matmul_readvariableop_resource:
ЮА3
 matmul_1_readvariableop_resource:	`А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
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
L:         `:         `:         `:         `*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         `2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         `2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         `2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         `2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         `2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         `2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         `2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         `2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         `2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         `2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         `2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         `2

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
@:         Ю:         `:         `: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         Ю
 
_user_specified_nameinputs:OK
'
_output_shapes
:         `
 
_user_specified_namestates:OK
'
_output_shapes
:         `
 
_user_specified_namestates
Л
З
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39100842

inputs

states
states_11
matmul_readvariableop_resource:	]°4
 matmul_1_readvariableop_resource:
Ю°.
biasadd_readvariableop_resource:	°
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         °2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2	
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
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ю2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ю2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ю2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ю2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ю2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ю2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ю2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ю2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

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
A:         ]:         Ю:         Ю: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         Ю
 
_user_specified_namestates:PL
(
_output_shapes
:         Ю
 
_user_specified_namestates
╨

э
lstm_21_while_cond_39103058,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3.
*lstm_21_while_less_lstm_21_strided_slice_1F
Blstm_21_while_lstm_21_while_cond_39103058___redundant_placeholder0F
Blstm_21_while_lstm_21_while_cond_39103058___redundant_placeholder1F
Blstm_21_while_lstm_21_while_cond_39103058___redundant_placeholder2F
Blstm_21_while_lstm_21_while_cond_39103058___redundant_placeholder3
lstm_21_while_identity
Ш
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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
э[
Ю
E__inference_lstm_21_layer_call_and_return_conditional_losses_39102472

inputs?
+lstm_cell_21_matmul_readvariableop_resource:
ЮА@
-lstm_cell_21_matmul_1_readvariableop_resource:	`А;
,lstm_cell_21_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_21/BiasAdd/ReadVariableOpв"lstm_cell_21/MatMul/ReadVariableOpв$lstm_cell_21/MatMul_1/ReadVariableOpвwhileD
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
value	B :`2
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
:         `2
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
value	B :`2
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
:         `2	
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
:         Ю2
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
valueB"    Ю   27
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
:         Ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOpн
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul╗
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOpй
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul_1а
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/add┤
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOpн
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dimє
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
lstm_cell_21/splitЖ
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/SigmoidК
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_1Л
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul}
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
lstm_cell_21/ReluЬ
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_1С
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/add_1К
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_2|
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/Relu_1а
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102388*
condR
while_cond_39102387*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         `*
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
:         `*
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
:         `2
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
:         `2

Identity╚
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Ю: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
у
═
while_cond_39104073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39104073___redundant_placeholder06
2while_while_cond_39104073___redundant_placeholder16
2while_while_cond_39104073___redundant_placeholder26
2while_while_cond_39104073___redundant_placeholder3
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
╙J
╘

lstm_21_while_body_39103059,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3+
'lstm_21_while_lstm_21_strided_slice_1_0g
clstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮАP
=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АK
<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	А
lstm_21_while_identity
lstm_21_while_identity_1
lstm_21_while_identity_2
lstm_21_while_identity_3
lstm_21_while_identity_4
lstm_21_while_identity_5)
%lstm_21_while_lstm_21_strided_slice_1e
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorM
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:
ЮАN
;lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:	`АI
:lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	АИв1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpв0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpв2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp╙
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2A
?lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0lstm_21_while_placeholderHlstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype023
1lstm_21/while/TensorArrayV2Read/TensorListGetItemт
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype022
0lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpў
!lstm_21/while/lstm_cell_21/MatMulMatMul8lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2#
!lstm_21/while/lstm_cell_21/MatMulч
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp=lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype024
2lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpр
#lstm_21/while/lstm_cell_21/MatMul_1MatMullstm_21_while_placeholder_2:lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2%
#lstm_21/while/lstm_cell_21/MatMul_1╪
lstm_21/while/lstm_cell_21/addAddV2+lstm_21/while/lstm_cell_21/MatMul:product:0-lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2 
lstm_21/while/lstm_cell_21/addр
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp<lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype023
1lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpх
"lstm_21/while/lstm_cell_21/BiasAddBiasAdd"lstm_21/while/lstm_cell_21/add:z:09lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2$
"lstm_21/while/lstm_cell_21/BiasAddЪ
*lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_21/while/lstm_cell_21/split/split_dimл
 lstm_21/while/lstm_cell_21/splitSplit3lstm_21/while/lstm_cell_21/split/split_dim:output:0+lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2"
 lstm_21/while/lstm_cell_21/split░
"lstm_21/while/lstm_cell_21/SigmoidSigmoid)lstm_21/while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2$
"lstm_21/while/lstm_cell_21/Sigmoid┤
$lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid)lstm_21/while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2&
$lstm_21/while/lstm_cell_21/Sigmoid_1└
lstm_21/while/lstm_cell_21/mulMul(lstm_21/while/lstm_cell_21/Sigmoid_1:y:0lstm_21_while_placeholder_3*
T0*'
_output_shapes
:         `2 
lstm_21/while/lstm_cell_21/mulз
lstm_21/while/lstm_cell_21/ReluRelu)lstm_21/while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2!
lstm_21/while/lstm_cell_21/Relu╘
 lstm_21/while/lstm_cell_21/mul_1Mul&lstm_21/while/lstm_cell_21/Sigmoid:y:0-lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2"
 lstm_21/while/lstm_cell_21/mul_1╔
 lstm_21/while/lstm_cell_21/add_1AddV2"lstm_21/while/lstm_cell_21/mul:z:0$lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2"
 lstm_21/while/lstm_cell_21/add_1┤
$lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid)lstm_21/while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2&
$lstm_21/while/lstm_cell_21/Sigmoid_2ж
!lstm_21/while/lstm_cell_21/Relu_1Relu$lstm_21/while/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2#
!lstm_21/while/lstm_cell_21/Relu_1╪
 lstm_21/while/lstm_cell_21/mul_2Mul(lstm_21/while/lstm_cell_21/Sigmoid_2:y:0/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2"
 lstm_21/while/lstm_cell_21/mul_2И
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
lstm_21/while/add/yЙ
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
lstm_21/while/add_1/yЮ
lstm_21/while/add_1AddV2(lstm_21_while_lstm_21_while_loop_counterlstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_21/while/add_1Л
lstm_21/while/IdentityIdentitylstm_21/while/add_1:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identityж
lstm_21/while/Identity_1Identity.lstm_21_while_lstm_21_while_maximum_iterations^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_1Н
lstm_21/while/Identity_2Identitylstm_21/while/add:z:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_2║
lstm_21/while/Identity_3IdentityBlstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_21/while/NoOp*
T0*
_output_shapes
: 2
lstm_21/while/Identity_3н
lstm_21/while/Identity_4Identity$lstm_21/while/lstm_cell_21/mul_2:z:0^lstm_21/while/NoOp*
T0*'
_output_shapes
:         `2
lstm_21/while/Identity_4н
lstm_21/while/Identity_5Identity$lstm_21/while/lstm_cell_21/add_1:z:0^lstm_21/while/NoOp*
T0*'
_output_shapes
:         `2
lstm_21/while/Identity_5Ж
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
9lstm_21_while_lstm_cell_21_matmul_readvariableop_resource;lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"╚
alstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensorclstm_21_while_tensorarrayv2read_tensorlistgetitem_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2f
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
хJ
╘

lstm_20_while_body_39103238,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3+
'lstm_20_while_lstm_20_strided_slice_1_0g
clstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0:	]°Q
=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°K
<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0:	°
lstm_20_while_identity
lstm_20_while_identity_1
lstm_20_while_identity_2
lstm_20_while_identity_3
lstm_20_while_identity_4
lstm_20_while_identity_5)
%lstm_20_while_lstm_20_strided_slice_1e
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorL
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource:	]°O
;lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°I
:lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpв0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpв2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp╙
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_20/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0lstm_20_while_placeholderHlstm_20/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_20/while/TensorArrayV2Read/TensorListGetItemс
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype022
0lstm_20/while/lstm_cell_20/MatMul/ReadVariableOpў
!lstm_20/while/lstm_cell_20/MatMulMatMul8lstm_20/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_20/while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2#
!lstm_20/while/lstm_cell_20/MatMulш
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp=lstm_20_while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype024
2lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOpр
#lstm_20/while/lstm_cell_20/MatMul_1MatMullstm_20_while_placeholder_2:lstm_20/while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2%
#lstm_20/while/lstm_cell_20/MatMul_1╪
lstm_20/while/lstm_cell_20/addAddV2+lstm_20/while/lstm_cell_20/MatMul:product:0-lstm_20/while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2 
lstm_20/while/lstm_cell_20/addр
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp<lstm_20_while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype023
1lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOpх
"lstm_20/while/lstm_cell_20/BiasAddBiasAdd"lstm_20/while/lstm_cell_20/add:z:09lstm_20/while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2$
"lstm_20/while/lstm_cell_20/BiasAddЪ
*lstm_20/while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_20/while/lstm_cell_20/split/split_dimп
 lstm_20/while/lstm_cell_20/splitSplit3lstm_20/while/lstm_cell_20/split/split_dim:output:0+lstm_20/while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2"
 lstm_20/while/lstm_cell_20/split▒
"lstm_20/while/lstm_cell_20/SigmoidSigmoid)lstm_20/while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2$
"lstm_20/while/lstm_cell_20/Sigmoid╡
$lstm_20/while/lstm_cell_20/Sigmoid_1Sigmoid)lstm_20/while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2&
$lstm_20/while/lstm_cell_20/Sigmoid_1┴
lstm_20/while/lstm_cell_20/mulMul(lstm_20/while/lstm_cell_20/Sigmoid_1:y:0lstm_20_while_placeholder_3*
T0*(
_output_shapes
:         Ю2 
lstm_20/while/lstm_cell_20/mulи
lstm_20/while/lstm_cell_20/ReluRelu)lstm_20/while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2!
lstm_20/while/lstm_cell_20/Relu╒
 lstm_20/while/lstm_cell_20/mul_1Mul&lstm_20/while/lstm_cell_20/Sigmoid:y:0-lstm_20/while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2"
 lstm_20/while/lstm_cell_20/mul_1╩
 lstm_20/while/lstm_cell_20/add_1AddV2"lstm_20/while/lstm_cell_20/mul:z:0$lstm_20/while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2"
 lstm_20/while/lstm_cell_20/add_1╡
$lstm_20/while/lstm_cell_20/Sigmoid_2Sigmoid)lstm_20/while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2&
$lstm_20/while/lstm_cell_20/Sigmoid_2з
!lstm_20/while/lstm_cell_20/Relu_1Relu$lstm_20/while/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2#
!lstm_20/while/lstm_cell_20/Relu_1┘
 lstm_20/while/lstm_cell_20/mul_2Mul(lstm_20/while/lstm_cell_20/Sigmoid_2:y:0/lstm_20/while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2"
 lstm_20/while/lstm_cell_20/mul_2И
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
lstm_20/while/add/yЙ
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
lstm_20/while/add_1/yЮ
lstm_20/while/add_1AddV2(lstm_20_while_lstm_20_while_loop_counterlstm_20/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_20/while/add_1Л
lstm_20/while/IdentityIdentitylstm_20/while/add_1:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identityж
lstm_20/while/Identity_1Identity.lstm_20_while_lstm_20_while_maximum_iterations^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_1Н
lstm_20/while/Identity_2Identitylstm_20/while/add:z:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_2║
lstm_20/while/Identity_3IdentityBlstm_20/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_20/while/NoOp*
T0*
_output_shapes
: 2
lstm_20/while/Identity_3о
lstm_20/while/Identity_4Identity$lstm_20/while/lstm_cell_20/mul_2:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:         Ю2
lstm_20/while/Identity_4о
lstm_20/while/Identity_5Identity$lstm_20/while/lstm_cell_20/add_1:z:0^lstm_20/while/NoOp*
T0*(
_output_shapes
:         Ю2
lstm_20/while/Identity_5Ж
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
9lstm_20_while_lstm_cell_20_matmul_readvariableop_resource;lstm_20_while_lstm_cell_20_matmul_readvariableop_resource_0"╚
alstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensorclstm_20_while_tensorarrayv2read_tensorlistgetitem_lstm_20_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2f
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
▒
╣
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102725

inputs#
lstm_20_39102703:	]°$
lstm_20_39102705:
Ю°
lstm_20_39102707:	°$
lstm_21_39102711:
ЮА#
lstm_21_39102713:	`А
lstm_21_39102715:	А#
dense_10_39102719:`
dense_10_39102721:
identityИв dense_10/StatefulPartitionedCallв"dropout_20/StatefulPartitionedCallв"dropout_21/StatefulPartitionedCallвlstm_20/StatefulPartitionedCallвlstm_21/StatefulPartitionedCallо
lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputslstm_20_39102703lstm_20_39102705lstm_20_39102707*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391026682!
lstm_20/StatefulPartitionedCallЫ
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391025012$
"dropout_20/StatefulPartitionedCall╥
lstm_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0lstm_21_39102711lstm_21_39102713lstm_21_39102715*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391024722!
lstm_21/StatefulPartitionedCall┐
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391023052$
"dropout_21/StatefulPartitionedCall├
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_10_39102719dense_10_39102721*
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391022492"
 dense_10/StatefulPartitionedCallИ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity 
NoOpNoOp!^dense_10/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
У
Й
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39104976

inputs
states_0
states_11
matmul_readvariableop_resource:	]°4
 matmul_1_readvariableop_resource:
Ю°.
biasadd_readvariableop_resource:	°
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         °2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2	
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
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ю2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ю2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ю2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ю2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ю2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ю2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ю2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ю2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

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
A:         ]:         Ю:         Ю: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ю
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ю
"
_user_specified_name
states/1
Е
f
H__inference_dropout_21_layer_call_and_return_conditional_losses_39102216

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:         `2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:         `2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
р
║
*__inference_lstm_20_layer_call_fn_39104169
inputs_0
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391007792
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  Ю2

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
У
Й
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105008

inputs
states_0
states_11
matmul_readvariableop_resource:	]°4
 matmul_1_readvariableop_resource:
Ю°.
biasadd_readvariableop_resource:	°
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         °2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2	
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
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         Ю2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         Ю2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         Ю2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         Ю2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         Ю2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         Ю2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         Ю2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         Ю2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ю2

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
A:         ]:         Ю:         Ю: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ю
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ю
"
_user_specified_name
states/1
Е
Ш
+__inference_dense_10_layer_call_fn_39104944

inputs
unknown:`
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391022492
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
:         `: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
░?
╘
while_body_39103923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]°G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°A
2while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив)while/lstm_cell_20/BiasAdd/ReadVariableOpв(while/lstm_cell_20/MatMul/ReadVariableOpв*while/lstm_cell_20/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp╫
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul╨
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOp└
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul_1╕
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/add╚
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOp┼
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/BiasAddК
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dimП
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
while/lstm_cell_20/splitЩ
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/SigmoidЭ
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_1б
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mulР
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu╡
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_1к
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/add_1Э
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_2П
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu_1╣
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2V
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_39103621
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_20_matmul_readvariableop_resource_0:	]°I
5while_lstm_cell_20_matmul_1_readvariableop_resource_0:
Ю°C
4while_lstm_cell_20_biasadd_readvariableop_resource_0:	°
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_20_matmul_readvariableop_resource:	]°G
3while_lstm_cell_20_matmul_1_readvariableop_resource:
Ю°A
2while_lstm_cell_20_biasadd_readvariableop_resource:	°Ив)while/lstm_cell_20/BiasAdd/ReadVariableOpв(while/lstm_cell_20/MatMul/ReadVariableOpв*while/lstm_cell_20/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_20/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_20_matmul_readvariableop_resource_0*
_output_shapes
:	]°*
dtype02*
(while/lstm_cell_20/MatMul/ReadVariableOp╫
while/lstm_cell_20/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul╨
*while/lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_20_matmul_1_readvariableop_resource_0* 
_output_shapes
:
Ю°*
dtype02,
*while/lstm_cell_20/MatMul_1/ReadVariableOp└
while/lstm_cell_20/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/MatMul_1╕
while/lstm_cell_20/addAddV2#while/lstm_cell_20/MatMul:product:0%while/lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/add╚
)while/lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_20_biasadd_readvariableop_resource_0*
_output_shapes	
:°*
dtype02+
)while/lstm_cell_20/BiasAdd/ReadVariableOp┼
while/lstm_cell_20/BiasAddBiasAddwhile/lstm_cell_20/add:z:01while/lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
while/lstm_cell_20/BiasAddК
"while/lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_20/split/split_dimП
while/lstm_cell_20/splitSplit+while/lstm_cell_20/split/split_dim:output:0#while/lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
while/lstm_cell_20/splitЩ
while/lstm_cell_20/SigmoidSigmoid!while/lstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/SigmoidЭ
while/lstm_cell_20/Sigmoid_1Sigmoid!while/lstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_1б
while/lstm_cell_20/mulMul while/lstm_cell_20/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mulР
while/lstm_cell_20/ReluRelu!while/lstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu╡
while/lstm_cell_20/mul_1Mulwhile/lstm_cell_20/Sigmoid:y:0%while/lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_1к
while/lstm_cell_20/add_1AddV2while/lstm_cell_20/mul:z:0while/lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/add_1Э
while/lstm_cell_20/Sigmoid_2Sigmoid!while/lstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Sigmoid_2П
while/lstm_cell_20/Relu_1Reluwhile/lstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/Relu_1╣
while/lstm_cell_20/mul_2Mul while/lstm_cell_20/Sigmoid_2:y:0'while/lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
while/lstm_cell_20/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_20/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_20/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         Ю2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         Ю:         Ю: : : : : 2V
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
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
: 
╥^
Ц
)sequential_10_lstm_21_while_body_39100509H
Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counterN
Jsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations+
'sequential_10_lstm_21_while_placeholder-
)sequential_10_lstm_21_while_placeholder_1-
)sequential_10_lstm_21_while_placeholder_2-
)sequential_10_lstm_21_while_placeholder_3G
Csequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1_0Г
sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0]
Isequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮА^
Ksequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АY
Jsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0:	А(
$sequential_10_lstm_21_while_identity*
&sequential_10_lstm_21_while_identity_1*
&sequential_10_lstm_21_while_identity_2*
&sequential_10_lstm_21_while_identity_3*
&sequential_10_lstm_21_while_identity_4*
&sequential_10_lstm_21_while_identity_5E
Asequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1Б
}sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor[
Gsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource:
ЮА\
Isequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource:	`АW
Hsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource:	АИв?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpв>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpв@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpя
Msequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2O
Msequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape╪
?sequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0'sequential_10_lstm_21_while_placeholderVsequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02A
?sequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItemМ
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOpIsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype02@
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOpп
/sequential_10/lstm_21/while/lstm_cell_21/MatMulMatMulFsequential_10/lstm_21/while/TensorArrayV2Read/TensorListGetItem:item:0Fsequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А21
/sequential_10/lstm_21/while/lstm_cell_21/MatMulС
@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOpKsequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype02B
@sequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOpШ
1sequential_10/lstm_21/while/lstm_cell_21/MatMul_1MatMul)sequential_10_lstm_21_while_placeholder_2Hsequential_10/lstm_21/while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А23
1sequential_10/lstm_21/while/lstm_cell_21/MatMul_1Р
,sequential_10/lstm_21/while/lstm_cell_21/addAddV29sequential_10/lstm_21/while/lstm_cell_21/MatMul:product:0;sequential_10/lstm_21/while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2.
,sequential_10/lstm_21/while/lstm_cell_21/addК
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOpJsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02A
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOpЭ
0sequential_10/lstm_21/while/lstm_cell_21/BiasAddBiasAdd0sequential_10/lstm_21/while/lstm_cell_21/add:z:0Gsequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А22
0sequential_10/lstm_21/while/lstm_cell_21/BiasAdd╢
8sequential_10/lstm_21/while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_10/lstm_21/while/lstm_cell_21/split/split_dimу
.sequential_10/lstm_21/while/lstm_cell_21/splitSplitAsequential_10/lstm_21/while/lstm_cell_21/split/split_dim:output:09sequential_10/lstm_21/while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split20
.sequential_10/lstm_21/while/lstm_cell_21/split┌
0sequential_10/lstm_21/while/lstm_cell_21/SigmoidSigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `22
0sequential_10/lstm_21/while/lstm_cell_21/Sigmoid▐
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1Sigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `24
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1°
,sequential_10/lstm_21/while/lstm_cell_21/mulMul6sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_1:y:0)sequential_10_lstm_21_while_placeholder_3*
T0*'
_output_shapes
:         `2.
,sequential_10/lstm_21/while/lstm_cell_21/mul╤
-sequential_10/lstm_21/while/lstm_cell_21/ReluRelu7sequential_10/lstm_21/while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2/
-sequential_10/lstm_21/while/lstm_cell_21/ReluМ
.sequential_10/lstm_21/while/lstm_cell_21/mul_1Mul4sequential_10/lstm_21/while/lstm_cell_21/Sigmoid:y:0;sequential_10/lstm_21/while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `20
.sequential_10/lstm_21/while/lstm_cell_21/mul_1Б
.sequential_10/lstm_21/while/lstm_cell_21/add_1AddV20sequential_10/lstm_21/while/lstm_cell_21/mul:z:02sequential_10/lstm_21/while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `20
.sequential_10/lstm_21/while/lstm_cell_21/add_1▐
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2Sigmoid7sequential_10/lstm_21/while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `24
2sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2╨
/sequential_10/lstm_21/while/lstm_cell_21/Relu_1Relu2sequential_10/lstm_21/while/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `21
/sequential_10/lstm_21/while/lstm_cell_21/Relu_1Р
.sequential_10/lstm_21/while/lstm_cell_21/mul_2Mul6sequential_10/lstm_21/while/lstm_cell_21/Sigmoid_2:y:0=sequential_10/lstm_21/while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `20
.sequential_10/lstm_21/while/lstm_cell_21/mul_2╬
@sequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_10_lstm_21_while_placeholder_1'sequential_10_lstm_21_while_placeholder2sequential_10/lstm_21/while/lstm_cell_21/mul_2:z:0*
_output_shapes
: *
element_dtype02B
@sequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItemИ
!sequential_10/lstm_21/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_10/lstm_21/while/add/y┴
sequential_10/lstm_21/while/addAddV2'sequential_10_lstm_21_while_placeholder*sequential_10/lstm_21/while/add/y:output:0*
T0*
_output_shapes
: 2!
sequential_10/lstm_21/while/addМ
#sequential_10/lstm_21/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#sequential_10/lstm_21/while/add_1/yф
!sequential_10/lstm_21/while/add_1AddV2Dsequential_10_lstm_21_while_sequential_10_lstm_21_while_loop_counter,sequential_10/lstm_21/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!sequential_10/lstm_21/while/add_1├
$sequential_10/lstm_21/while/IdentityIdentity%sequential_10/lstm_21/while/add_1:z:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_10/lstm_21/while/Identityь
&sequential_10/lstm_21/while/Identity_1IdentityJsequential_10_lstm_21_while_sequential_10_lstm_21_while_maximum_iterations!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_1┼
&sequential_10/lstm_21/while/Identity_2Identity#sequential_10/lstm_21/while/add:z:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_2Є
&sequential_10/lstm_21/while/Identity_3IdentityPsequential_10/lstm_21/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_10/lstm_21/while/NoOp*
T0*
_output_shapes
: 2(
&sequential_10/lstm_21/while/Identity_3х
&sequential_10/lstm_21/while/Identity_4Identity2sequential_10/lstm_21/while/lstm_cell_21/mul_2:z:0!^sequential_10/lstm_21/while/NoOp*
T0*'
_output_shapes
:         `2(
&sequential_10/lstm_21/while/Identity_4х
&sequential_10/lstm_21/while/Identity_5Identity2sequential_10/lstm_21/while/lstm_cell_21/add_1:z:0!^sequential_10/lstm_21/while/NoOp*
T0*'
_output_shapes
:         `2(
&sequential_10/lstm_21/while/Identity_5╠
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
&sequential_10_lstm_21_while_identity_5/sequential_10/lstm_21/while/Identity_5:output:0"Ц
Hsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resourceJsequential_10_lstm_21_while_lstm_cell_21_biasadd_readvariableop_resource_0"Ш
Isequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resourceKsequential_10_lstm_21_while_lstm_cell_21_matmul_1_readvariableop_resource_0"Ф
Gsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resourceIsequential_10_lstm_21_while_lstm_cell_21_matmul_readvariableop_resource_0"И
Asequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1Csequential_10_lstm_21_while_sequential_10_lstm_21_strided_slice_1_0"А
}sequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensorsequential_10_lstm_21_while_tensorarrayv2read_tensorlistgetitem_sequential_10_lstm_21_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2В
?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp?sequential_10/lstm_21/while/lstm_cell_21/BiasAdd/ReadVariableOp2А
>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp>sequential_10/lstm_21/while/lstm_cell_21/MatMul/ReadVariableOp2Д
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
¤%
є
while_body_39101550
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_21_39101574_0:
ЮА0
while_lstm_cell_21_39101576_0:	`А,
while_lstm_cell_21_39101578_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_21_39101574:
ЮА.
while_lstm_cell_21_39101576:	`А*
while_lstm_cell_21_39101578:	АИв*while/lstm_cell_21/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemщ
*while/lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_21_39101574_0while_lstm_cell_21_39101576_0while_lstm_cell_21_39101578_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         `:         `:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391014722,
*while/lstm_cell_21/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3д
while/Identity_4Identity3while/lstm_cell_21/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_4д
while/Identity_5Identity3while/lstm_cell_21/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_5З

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
while_lstm_cell_21_39101574while_lstm_cell_21_39101574_0"<
while_lstm_cell_21_39101576while_lstm_cell_21_39101576_0"<
while_lstm_cell_21_39101578while_lstm_cell_21_39101578_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2X
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_39101953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39101953___redundant_placeholder06
2while_while_cond_39101953___redundant_placeholder16
2while_while_cond_39101953___redundant_placeholder26
2while_while_cond_39101953___redundant_placeholder3
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
╪
I
-__inference_dropout_20_layer_call_fn_39104224

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
:         Ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391020512
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         Ю2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ю:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
├\
а
E__inference_lstm_20_layer_call_and_return_conditional_losses_39103856
inputs_0>
+lstm_cell_20_matmul_readvariableop_resource:	]°A
-lstm_cell_20_matmul_1_readvariableop_resource:
Ю°;
,lstm_cell_20_biasadd_readvariableop_resource:	°
identityИв#lstm_cell_20/BiasAdd/ReadVariableOpв"lstm_cell_20/MatMul/ReadVariableOpв$lstm_cell_20/MatMul_1/ReadVariableOpвwhileF
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
B :Ю2
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
B :Ю2
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
:         Ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
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
B :Ю2
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
:         Ю2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOpн
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul╝
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOpй
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul_1а
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/add┤
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOpн
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dimў
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
lstm_cell_20/splitЗ
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/SigmoidЛ
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_1М
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/ReluЭ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_1Т
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/add_1Л
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Relu_1б
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39103772*
condR
while_cond_39103771*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  Ю*
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
:         Ю*
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
!:                  Ю2
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
!:                  Ю2

Identity╚
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
╘
I
-__inference_dropout_21_layer_call_fn_39104899

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
:         `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391022162
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         `2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         `:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
╟
∙
/__inference_lstm_cell_20_layer_call_fn_39105042

inputs
states_0
states_1
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
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
<:         Ю:         Ю:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_391008422
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ю2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         Ю2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         Ю2

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
A:         ]:         Ю:         Ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         Ю
"
_user_specified_name
states/0:RN
(
_output_shapes
:         Ю
"
_user_specified_name
states/1
Б
Й
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105106

inputs
states_0
states_12
matmul_readvariableop_resource:
ЮА3
 matmul_1_readvariableop_resource:	`А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
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
L:         `:         `:         `:         `*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         `2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         `2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         `2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         `2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         `2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         `2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         `2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         `2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         `2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         `2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         `2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         `2

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
@:         Ю:         `:         `: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         Ю
 
_user_specified_nameinputs:QM
'
_output_shapes
:         `
"
_user_specified_name
states/0:QM
'
_output_shapes
:         `
"
_user_specified_name
states/1
╨

э
lstm_21_while_cond_39103392,
(lstm_21_while_lstm_21_while_loop_counter2
.lstm_21_while_lstm_21_while_maximum_iterations
lstm_21_while_placeholder
lstm_21_while_placeholder_1
lstm_21_while_placeholder_2
lstm_21_while_placeholder_3.
*lstm_21_while_less_lstm_21_strided_slice_1F
Blstm_21_while_lstm_21_while_cond_39103392___redundant_placeholder0F
Blstm_21_while_lstm_21_while_cond_39103392___redundant_placeholder1F
Blstm_21_while_lstm_21_while_cond_39103392___redundant_placeholder2F
Blstm_21_while_lstm_21_while_cond_39103392___redundant_placeholder3
lstm_21_while_identity
Ш
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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
ШС
К
$__inference__traced_restore_39105371
file_prefix2
 assignvariableop_dense_10_kernel:`.
 assignvariableop_1_dense_10_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_20_lstm_cell_20_kernel:	]°L
8assignvariableop_8_lstm_20_lstm_cell_20_recurrent_kernel:
Ю°;
,assignvariableop_9_lstm_20_lstm_cell_20_bias:	°C
/assignvariableop_10_lstm_21_lstm_cell_21_kernel:
ЮАL
9assignvariableop_11_lstm_21_lstm_cell_21_recurrent_kernel:	`А<
-assignvariableop_12_lstm_21_lstm_cell_21_bias:	А#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: <
*assignvariableop_17_adam_dense_10_kernel_m:`6
(assignvariableop_18_adam_dense_10_bias_m:I
6assignvariableop_19_adam_lstm_20_lstm_cell_20_kernel_m:	]°T
@assignvariableop_20_adam_lstm_20_lstm_cell_20_recurrent_kernel_m:
Ю°C
4assignvariableop_21_adam_lstm_20_lstm_cell_20_bias_m:	°J
6assignvariableop_22_adam_lstm_21_lstm_cell_21_kernel_m:
ЮАS
@assignvariableop_23_adam_lstm_21_lstm_cell_21_recurrent_kernel_m:	`АC
4assignvariableop_24_adam_lstm_21_lstm_cell_21_bias_m:	А<
*assignvariableop_25_adam_dense_10_kernel_v:`6
(assignvariableop_26_adam_dense_10_bias_v:I
6assignvariableop_27_adam_lstm_20_lstm_cell_20_kernel_v:	]°T
@assignvariableop_28_adam_lstm_20_lstm_cell_20_recurrent_kernel_v:
Ю°C
4assignvariableop_29_adam_lstm_20_lstm_cell_20_bias_v:	°J
6assignvariableop_30_adam_lstm_21_lstm_cell_21_kernel_v:
ЮАS
@assignvariableop_31_adam_lstm_21_lstm_cell_21_recurrent_kernel_v:	`АC
4assignvariableop_32_adam_lstm_21_lstm_cell_21_bias_v:	А
identity_34ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9╛
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*╩
value└B╜"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╥
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices╪
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

IdentityЯ
AssignVariableOpAssignVariableOp assignvariableop_dense_10_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1е
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_10_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2б
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4г
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5в
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6к
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7│
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_20_lstm_cell_20_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╜
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_20_lstm_cell_20_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9▒
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_20_lstm_cell_20_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╖
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_21_lstm_cell_21_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┴
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_21_lstm_cell_21_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╡
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_21_lstm_cell_21_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13б
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14б
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15г
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16г
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17▓
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_10_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18░
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_10_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╛
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_20_lstm_cell_20_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╚
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_20_lstm_cell_20_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╝
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_20_lstm_cell_20_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╛
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_lstm_21_lstm_cell_21_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╚
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_lstm_21_lstm_cell_21_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╝
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_21_lstm_cell_21_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▓
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_10_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26░
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_10_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╛
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_20_lstm_cell_20_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╚
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_lstm_20_lstm_cell_20_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╝
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_20_lstm_cell_20_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╛
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_21_lstm_cell_21_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╚
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_21_lstm_cell_21_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╝
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_21_lstm_cell_21_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp┤
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33f
Identity_34IdentityIdentity_33:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_34Ь
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
у
═
while_cond_39103771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39103771___redundant_placeholder06
2while_while_cond_39103771___redundant_placeholder16
2while_while_cond_39103771___redundant_placeholder26
2while_while_cond_39103771___redundant_placeholder3
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
Д\
Ю
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104158

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	]°A
-lstm_cell_20_matmul_1_readvariableop_resource:
Ю°;
,lstm_cell_20_biasadd_readvariableop_resource:	°
identityИв#lstm_cell_20/BiasAdd/ReadVariableOpв"lstm_cell_20/MatMul/ReadVariableOpв$lstm_cell_20/MatMul_1/ReadVariableOpвwhileD
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
B :Ю2
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
B :Ю2
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
:         Ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
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
B :Ю2
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
:         Ю2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOpн
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul╝
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOpй
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul_1а
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/add┤
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOpн
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dimў
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
lstm_cell_20/splitЗ
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/SigmoidЛ
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_1М
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/ReluЭ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_1Т
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/add_1Л
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Relu_1б
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104074*
condR
while_cond_39104073*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ю*
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
:         Ю*
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
:         Ю2
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
:         Ю2

Identity╚
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╛F
О
E__inference_lstm_21_layer_call_and_return_conditional_losses_39101619

inputs)
lstm_cell_21_39101537:
ЮА(
lstm_cell_21_39101539:	`А$
lstm_cell_21_39101541:	А
identityИв$lstm_cell_21/StatefulPartitionedCallвwhileD
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
value	B :`2
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
:         `2
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
value	B :`2
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
:         `2	
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
!:                  Ю2
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
valueB"    Ю   27
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
:         Ю*
shrink_axis_mask2
strided_slice_2е
$lstm_cell_21/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_21_39101537lstm_cell_21_39101539lstm_cell_21_39101541*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         `:         `:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_391014722&
$lstm_cell_21/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_21_39101537lstm_cell_21_39101539lstm_cell_21_39101541*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39101550*
condR
while_cond_39101549*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0TensorArrayV2Stack/TensorListStack/element_shapeё
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  `*
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
:         `*
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
 :                  `2
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
 :                  `2

Identity}
NoOpNoOp%^lstm_cell_21/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Ю: : : 2L
$lstm_cell_21/StatefulPartitionedCall$lstm_cell_21/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  Ю
 
_user_specified_nameinputs
Ю?
╘
while_body_39104598
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮАH
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ЮАF
3while_lstm_cell_21_matmul_1_readvariableop_resource:	`АA
2while_lstm_cell_21_biasadd_readvariableop_resource:	АИв)while/lstm_cell_21/BiasAdd/ReadVariableOpв(while/lstm_cell_21/MatMul/ReadVariableOpв*while/lstm_cell_21/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp╫
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul╧
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOp└
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul_1╕
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/add╚
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOp┼
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/BiasAddК
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dimЛ
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
while/lstm_cell_21/splitШ
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/SigmoidЬ
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_1а
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mulП
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu┤
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_1й
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/add_1Ь
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_2О
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu_1╕
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2V
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
▀
═
while_cond_39101549
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39101549___redundant_placeholder06
2while_while_cond_39101549___redundant_placeholder16
2while_while_cond_39101549___redundant_placeholder26
2while_while_cond_39101549___redundant_placeholder3
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
@: : : : :         `:         `: ::::: 

_output_shapes
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
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
:
э[
Ю
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104833

inputs?
+lstm_cell_21_matmul_readvariableop_resource:
ЮА@
-lstm_cell_21_matmul_1_readvariableop_resource:	`А;
,lstm_cell_21_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_21/BiasAdd/ReadVariableOpв"lstm_cell_21/MatMul/ReadVariableOpв$lstm_cell_21/MatMul_1/ReadVariableOpвwhileD
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
value	B :`2
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
:         `2
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
value	B :`2
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
:         `2	
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
:         Ю2
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
valueB"    Ю   27
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
:         Ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOpн
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul╗
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOpй
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul_1а
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/add┤
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOpн
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dimє
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
lstm_cell_21/splitЖ
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/SigmoidК
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_1Л
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul}
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
lstm_cell_21/ReluЬ
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_1С
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/add_1К
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_2|
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/Relu_1а
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39104749*
condR
while_cond_39104748*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         `*
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
:         `*
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
:         `2
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
:         `2

Identity╚
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Ю: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
╫
g
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104219

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
:         Ю2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         Ю*
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
:         Ю2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         Ю2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         Ю2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         Ю2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ю:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
д
я
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102256

inputs#
lstm_20_39102039:	]°$
lstm_20_39102041:
Ю°
lstm_20_39102043:	°$
lstm_21_39102204:
ЮА#
lstm_21_39102206:	`А
lstm_21_39102208:	А#
dense_10_39102250:`
dense_10_39102252:
identityИв dense_10/StatefulPartitionedCallвlstm_20/StatefulPartitionedCallвlstm_21/StatefulPartitionedCallо
lstm_20/StatefulPartitionedCallStatefulPartitionedCallinputslstm_20_39102039lstm_20_39102041lstm_20_39102043*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391020382!
lstm_20/StatefulPartitionedCallГ
dropout_20/PartitionedCallPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391020512
dropout_20/PartitionedCall╩
lstm_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0lstm_21_39102204lstm_21_39102206lstm_21_39102208*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391022032!
lstm_21/StatefulPartitionedCallВ
dropout_21/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391022162
dropout_21/PartitionedCall╗
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0dense_10_39102250dense_10_39102252*
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391022492"
 dense_10/StatefulPartitionedCallИ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╡
NoOpNoOp!^dense_10/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
м

╥
0__inference_sequential_10_layer_call_fn_39102275
lstm_20_input
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
	unknown_2:
ЮА
	unknown_3:	`А
	unknown_4:	А
	unknown_5:`
	unknown_6:
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_391022562
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
_user_specified_namelstm_20_input
Ю?
╘
while_body_39104447
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮАH
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ЮАF
3while_lstm_cell_21_matmul_1_readvariableop_resource:	`АA
2while_lstm_cell_21_biasadd_readvariableop_resource:	АИв)while/lstm_cell_21/BiasAdd/ReadVariableOpв(while/lstm_cell_21/MatMul/ReadVariableOpв*while/lstm_cell_21/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp╫
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul╧
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOp└
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul_1╕
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/add╚
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOp┼
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/BiasAddК
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dimЛ
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
while/lstm_cell_21/splitШ
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/SigmoidЬ
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_1а
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mulП
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu┤
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_1й
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/add_1Ь
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_2О
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu_1╕
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2V
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
Ю?
╘
while_body_39104749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮАH
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ЮАF
3while_lstm_cell_21_matmul_1_readvariableop_resource:	`АA
2while_lstm_cell_21_biasadd_readvariableop_resource:	АИв)while/lstm_cell_21/BiasAdd/ReadVariableOpв(while/lstm_cell_21/MatMul/ReadVariableOpв*while/lstm_cell_21/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp╫
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul╧
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOp└
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul_1╕
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/add╚
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOp┼
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/BiasAddК
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dimЛ
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
while/lstm_cell_21/splitШ
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/SigmoidЬ
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_1а
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mulП
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu┤
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_1й
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/add_1Ь
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_2О
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu_1╕
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2V
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
╗
f
-__inference_dropout_20_layer_call_fn_39104229

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
:         Ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391025012
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ю2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         Ю22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
м

╥
0__inference_sequential_10_layer_call_fn_39102765
lstm_20_input
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
	unknown_2:
ЮА
	unknown_3:	`А
	unknown_4:	А
	unknown_5:`
	unknown_6:
identityИвStatefulPartitionedCall╘
StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_391027252
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
_user_specified_namelstm_20_input
╨!
¤
F__inference_dense_10_layer_call_and_return_conditional_losses_39104935

inputs3
!tensordot_readvariableop_resource:`-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:`*
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
:         `2
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
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
╘

э
lstm_20_while_cond_39102910,
(lstm_20_while_lstm_20_while_loop_counter2
.lstm_20_while_lstm_20_while_maximum_iterations
lstm_20_while_placeholder
lstm_20_while_placeholder_1
lstm_20_while_placeholder_2
lstm_20_while_placeholder_3.
*lstm_20_while_less_lstm_20_strided_slice_1F
Blstm_20_while_lstm_20_while_cond_39102910___redundant_placeholder0F
Blstm_20_while_lstm_20_while_cond_39102910___redundant_placeholder1F
Blstm_20_while_lstm_20_while_cond_39102910___redundant_placeholder2F
Blstm_20_while_lstm_20_while_cond_39102910___redundant_placeholder3
lstm_20_while_identity
Ш
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
╡
╕
*__inference_lstm_21_layer_call_fn_39104877

inputs
unknown:
ЮА
	unknown_0:	`А
	unknown_1:	А
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391024722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
Ч

╦
0__inference_sequential_10_layer_call_fn_39103554

inputs
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
	unknown_2:
ЮА
	unknown_3:	`А
	unknown_4:	А
	unknown_5:`
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_391027252
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
∙
З
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39101472

inputs

states
states_12
matmul_readvariableop_resource:
ЮА3
 matmul_1_readvariableop_resource:	`А.
biasadd_readvariableop_resource:	А
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
MatMulФ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         А2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2	
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
L:         `:         `:         `:         `*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         `2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         `2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         `2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:         `2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         `2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         `2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         `2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         `2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         `2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         `2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:         `2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:         `2

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
@:         Ю:         `:         `: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         Ю
 
_user_specified_nameinputs:OK
'
_output_shapes
:         `
 
_user_specified_namestates:OK
'
_output_shapes
:         `
 
_user_specified_namestates
Ю?
╘
while_body_39102119
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮАH
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ЮАF
3while_lstm_cell_21_matmul_1_readvariableop_resource:	`АA
2while_lstm_cell_21_biasadd_readvariableop_resource:	АИв)while/lstm_cell_21/BiasAdd/ReadVariableOpв(while/lstm_cell_21/MatMul/ReadVariableOpв*while/lstm_cell_21/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp╫
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul╧
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOp└
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul_1╕
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/add╚
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOp┼
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/BiasAddК
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dimЛ
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
while/lstm_cell_21/splitШ
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/SigmoidЬ
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_1а
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mulП
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu┤
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_1й
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/add_1Ь
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_2О
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu_1╕
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2V
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
э[
Ю
E__inference_lstm_21_layer_call_and_return_conditional_losses_39102203

inputs?
+lstm_cell_21_matmul_readvariableop_resource:
ЮА@
-lstm_cell_21_matmul_1_readvariableop_resource:	`А;
,lstm_cell_21_biasadd_readvariableop_resource:	А
identityИв#lstm_cell_21/BiasAdd/ReadVariableOpв"lstm_cell_21/MatMul/ReadVariableOpв$lstm_cell_21/MatMul_1/ReadVariableOpвwhileD
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
value	B :`2
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
:         `2
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
value	B :`2
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
:         `2	
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
:         Ю2
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
valueB"    Ю   27
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
:         Ю*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_21/MatMul/ReadVariableOpReadVariableOp+lstm_cell_21_matmul_readvariableop_resource* 
_output_shapes
:
ЮА*
dtype02$
"lstm_cell_21/MatMul/ReadVariableOpн
lstm_cell_21/MatMulMatMulstrided_slice_2:output:0*lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul╗
$lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_21_matmul_1_readvariableop_resource*
_output_shapes
:	`А*
dtype02&
$lstm_cell_21/MatMul_1/ReadVariableOpй
lstm_cell_21/MatMul_1MatMulzeros:output:0,lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/MatMul_1а
lstm_cell_21/addAddV2lstm_cell_21/MatMul:product:0lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/add┤
#lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_21_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype02%
#lstm_cell_21/BiasAdd/ReadVariableOpн
lstm_cell_21/BiasAddBiasAddlstm_cell_21/add:z:0+lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
lstm_cell_21/BiasAdd~
lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_21/split/split_dimє
lstm_cell_21/splitSplit%lstm_cell_21/split/split_dim:output:0lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
lstm_cell_21/splitЖ
lstm_cell_21/SigmoidSigmoidlstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/SigmoidК
lstm_cell_21/Sigmoid_1Sigmoidlstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_1Л
lstm_cell_21/mulMullstm_cell_21/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul}
lstm_cell_21/ReluRelulstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
lstm_cell_21/ReluЬ
lstm_cell_21/mul_1Mullstm_cell_21/Sigmoid:y:0lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_1С
lstm_cell_21/add_1AddV2lstm_cell_21/mul:z:0lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/add_1К
lstm_cell_21/Sigmoid_2Sigmoidlstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
lstm_cell_21/Sigmoid_2|
lstm_cell_21/Relu_1Relulstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/Relu_1а
lstm_cell_21/mul_2Mullstm_cell_21/Sigmoid_2:y:0!lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
lstm_cell_21/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_21_matmul_readvariableop_resource-lstm_cell_21_matmul_1_readvariableop_resource,lstm_cell_21_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         `:         `: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39102119*
condR
while_cond_39102118*K
output_shapes:
8: : : : :         `:         `: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    `   22
0TensorArrayV2Stack/TensorListStack/element_shapeш
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         `*
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
:         `*
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
:         `2
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
:         `2

Identity╚
NoOpNoOp$^lstm_cell_21/BiasAdd/ReadVariableOp#^lstm_cell_21/MatMul/ReadVariableOp%^lstm_cell_21/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Ю: : : 2J
#lstm_cell_21/BiasAdd/ReadVariableOp#lstm_cell_21/BiasAdd/ReadVariableOp2H
"lstm_cell_21/MatMul/ReadVariableOp"lstm_cell_21/MatMul/ReadVariableOp2L
$lstm_cell_21/MatMul_1/ReadVariableOp$lstm_cell_21/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
╨!
¤
F__inference_dense_10_layer_call_and_return_conditional_losses_39102249

inputs3
!tensordot_readvariableop_resource:`-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЦ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:`*
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
:         `2
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
:         `: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         `
 
_user_specified_nameinputs
р
║
*__inference_lstm_21_layer_call_fn_39104844
inputs_0
unknown:
ЮА
	unknown_0:	`А
	unknown_1:	А
identityИвStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391014092
StatefulPartitionedCallИ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  `2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  Ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  Ю
"
_user_specified_name
inputs/0
╡
╕
*__inference_lstm_21_layer_call_fn_39104866

inputs
unknown:
ЮА
	unknown_0:	`А
	unknown_1:	А
identityИвStatefulPartitionedCallЖ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391022032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         Ю: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ю
 
_user_specified_nameinputs
╢
╕
*__inference_lstm_20_layer_call_fn_39104202

inputs
unknown:	]°
	unknown_0:
Ю°
	unknown_1:	°
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391026682
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         Ю2

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
Ю?
╘
while_body_39102388
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_21_matmul_readvariableop_resource_0:
ЮАH
5while_lstm_cell_21_matmul_1_readvariableop_resource_0:	`АC
4while_lstm_cell_21_biasadd_readvariableop_resource_0:	А
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_21_matmul_readvariableop_resource:
ЮАF
3while_lstm_cell_21_matmul_1_readvariableop_resource:	`АA
2while_lstm_cell_21_biasadd_readvariableop_resource:	АИв)while/lstm_cell_21/BiasAdd/ReadVariableOpв(while/lstm_cell_21/MatMul/ReadVariableOpв*while/lstm_cell_21/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         Ю*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_21/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_21_matmul_readvariableop_resource_0* 
_output_shapes
:
ЮА*
dtype02*
(while/lstm_cell_21/MatMul/ReadVariableOp╫
while/lstm_cell_21/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul╧
*while/lstm_cell_21/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_21_matmul_1_readvariableop_resource_0*
_output_shapes
:	`А*
dtype02,
*while/lstm_cell_21/MatMul_1/ReadVariableOp└
while/lstm_cell_21/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_21/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/MatMul_1╕
while/lstm_cell_21/addAddV2#while/lstm_cell_21/MatMul:product:0%while/lstm_cell_21/MatMul_1:product:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/add╚
)while/lstm_cell_21/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_21_biasadd_readvariableop_resource_0*
_output_shapes	
:А*
dtype02+
)while/lstm_cell_21/BiasAdd/ReadVariableOp┼
while/lstm_cell_21/BiasAddBiasAddwhile/lstm_cell_21/add:z:01while/lstm_cell_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А2
while/lstm_cell_21/BiasAddК
"while/lstm_cell_21/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_21/split/split_dimЛ
while/lstm_cell_21/splitSplit+while/lstm_cell_21/split/split_dim:output:0#while/lstm_cell_21/BiasAdd:output:0*
T0*`
_output_shapesN
L:         `:         `:         `:         `*
	num_split2
while/lstm_cell_21/splitШ
while/lstm_cell_21/SigmoidSigmoid!while/lstm_cell_21/split:output:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/SigmoidЬ
while/lstm_cell_21/Sigmoid_1Sigmoid!while/lstm_cell_21/split:output:1*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_1а
while/lstm_cell_21/mulMul while/lstm_cell_21/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mulП
while/lstm_cell_21/ReluRelu!while/lstm_cell_21/split:output:2*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu┤
while/lstm_cell_21/mul_1Mulwhile/lstm_cell_21/Sigmoid:y:0%while/lstm_cell_21/Relu:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_1й
while/lstm_cell_21/add_1AddV2while/lstm_cell_21/mul:z:0while/lstm_cell_21/mul_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/add_1Ь
while/lstm_cell_21/Sigmoid_2Sigmoid!while/lstm_cell_21/split:output:3*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Sigmoid_2О
while/lstm_cell_21/Relu_1Reluwhile/lstm_cell_21/add_1:z:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/Relu_1╕
while/lstm_cell_21/mul_2Mul while/lstm_cell_21/Sigmoid_2:y:0'while/lstm_cell_21/Relu_1:activations:0*
T0*'
_output_shapes
:         `2
while/lstm_cell_21/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3Н
while/Identity_4Identitywhile/lstm_cell_21/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_4Н
while/Identity_5Identitywhile/lstm_cell_21/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:         `2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         `:         `: : : : : 2V
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
: :-)
'
_output_shapes
:         `:-)
'
_output_shapes
:         `:

_output_shapes
: :

_output_shapes
: 
╣
Ў
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102790
lstm_20_input#
lstm_20_39102768:	]°$
lstm_20_39102770:
Ю°
lstm_20_39102772:	°$
lstm_21_39102776:
ЮА#
lstm_21_39102778:	`А
lstm_21_39102780:	А#
dense_10_39102784:`
dense_10_39102786:
identityИв dense_10/StatefulPartitionedCallвlstm_20/StatefulPartitionedCallвlstm_21/StatefulPartitionedCall╡
lstm_20/StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputlstm_20_39102768lstm_20_39102770lstm_20_39102772*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391020382!
lstm_20/StatefulPartitionedCallГ
dropout_20/PartitionedCallPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391020512
dropout_20/PartitionedCall╩
lstm_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_20/PartitionedCall:output:0lstm_21_39102776lstm_21_39102778lstm_21_39102780*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391022032!
lstm_21/StatefulPartitionedCallВ
dropout_21/PartitionedCallPartitionedCall(lstm_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391022162
dropout_21/PartitionedCall╗
 dense_10/StatefulPartitionedCallStatefulPartitionedCall#dropout_21/PartitionedCall:output:0dense_10_39102784dense_10_39102786*
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391022492"
 dense_10/StatefulPartitionedCallИ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity╡
NoOpNoOp!^dense_10/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_20_input
Д\
Ю
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104007

inputs>
+lstm_cell_20_matmul_readvariableop_resource:	]°A
-lstm_cell_20_matmul_1_readvariableop_resource:
Ю°;
,lstm_cell_20_biasadd_readvariableop_resource:	°
identityИв#lstm_cell_20/BiasAdd/ReadVariableOpв"lstm_cell_20/MatMul/ReadVariableOpв$lstm_cell_20/MatMul_1/ReadVariableOpвwhileD
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
B :Ю2
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
B :Ю2
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
:         Ю2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :Ю2
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
B :Ю2
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
:         Ю2	
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
"lstm_cell_20/MatMul/ReadVariableOpReadVariableOp+lstm_cell_20_matmul_readvariableop_resource*
_output_shapes
:	]°*
dtype02$
"lstm_cell_20/MatMul/ReadVariableOpн
lstm_cell_20/MatMulMatMulstrided_slice_2:output:0*lstm_cell_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul╝
$lstm_cell_20/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_20_matmul_1_readvariableop_resource* 
_output_shapes
:
Ю°*
dtype02&
$lstm_cell_20/MatMul_1/ReadVariableOpй
lstm_cell_20/MatMul_1MatMulzeros:output:0,lstm_cell_20/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/MatMul_1а
lstm_cell_20/addAddV2lstm_cell_20/MatMul:product:0lstm_cell_20/MatMul_1:product:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/add┤
#lstm_cell_20/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_20_biasadd_readvariableop_resource*
_output_shapes	
:°*
dtype02%
#lstm_cell_20/BiasAdd/ReadVariableOpн
lstm_cell_20/BiasAddBiasAddlstm_cell_20/add:z:0+lstm_cell_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         °2
lstm_cell_20/BiasAdd~
lstm_cell_20/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_20/split/split_dimў
lstm_cell_20/splitSplit%lstm_cell_20/split/split_dim:output:0lstm_cell_20/BiasAdd:output:0*
T0*d
_output_shapesR
P:         Ю:         Ю:         Ю:         Ю*
	num_split2
lstm_cell_20/splitЗ
lstm_cell_20/SigmoidSigmoidlstm_cell_20/split:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/SigmoidЛ
lstm_cell_20/Sigmoid_1Sigmoidlstm_cell_20/split:output:1*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_1М
lstm_cell_20/mulMullstm_cell_20/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul~
lstm_cell_20/ReluRelulstm_cell_20/split:output:2*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/ReluЭ
lstm_cell_20/mul_1Mullstm_cell_20/Sigmoid:y:0lstm_cell_20/Relu:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_1Т
lstm_cell_20/add_1AddV2lstm_cell_20/mul:z:0lstm_cell_20/mul_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/add_1Л
lstm_cell_20/Sigmoid_2Sigmoidlstm_cell_20/split:output:3*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Sigmoid_2}
lstm_cell_20/Relu_1Relulstm_cell_20/add_1:z:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/Relu_1б
lstm_cell_20/mul_2Mullstm_cell_20/Sigmoid_2:y:0!lstm_cell_20/Relu_1:activations:0*
T0*(
_output_shapes
:         Ю2
lstm_cell_20/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_20_matmul_readvariableop_resource-lstm_cell_20_matmul_1_readvariableop_resource,lstm_cell_20_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         Ю:         Ю: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_39103923*
condR
while_cond_39103922*M
output_shapes<
:: : : : :         Ю:         Ю: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    Ю   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         Ю*
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
:         Ю*
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
:         Ю2
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
:         Ю2

Identity╚
NoOpNoOp$^lstm_cell_20/BiasAdd/ReadVariableOp#^lstm_cell_20/MatMul/ReadVariableOp%^lstm_cell_20/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_20/BiasAdd/ReadVariableOp#lstm_cell_20/BiasAdd/ReadVariableOp2H
"lstm_cell_20/MatMul/ReadVariableOp"lstm_cell_20/MatMul/ReadVariableOp2L
$lstm_cell_20/MatMul_1/ReadVariableOp$lstm_cell_20/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
у
═
while_cond_39103922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_39103922___redundant_placeholder06
2while_while_cond_39103922___redundant_placeholder16
2while_while_cond_39103922___redundant_placeholder26
2while_while_cond_39103922___redundant_placeholder3
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
B: : : : :         Ю:         Ю: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         Ю:.*
(
_output_shapes
:         Ю:

_output_shapes
: :

_output_shapes
:
╞
└
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102815
lstm_20_input#
lstm_20_39102793:	]°$
lstm_20_39102795:
Ю°
lstm_20_39102797:	°$
lstm_21_39102801:
ЮА#
lstm_21_39102803:	`А
lstm_21_39102805:	А#
dense_10_39102809:`
dense_10_39102811:
identityИв dense_10/StatefulPartitionedCallв"dropout_20/StatefulPartitionedCallв"dropout_21/StatefulPartitionedCallвlstm_20/StatefulPartitionedCallвlstm_21/StatefulPartitionedCall╡
lstm_20/StatefulPartitionedCallStatefulPartitionedCalllstm_20_inputlstm_20_39102793lstm_20_39102795lstm_20_39102797*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_20_layer_call_and_return_conditional_losses_391026682!
lstm_20/StatefulPartitionedCallЫ
"dropout_20/StatefulPartitionedCallStatefulPartitionedCall(lstm_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ю* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_20_layer_call_and_return_conditional_losses_391025012$
"dropout_20/StatefulPartitionedCall╥
lstm_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_20/StatefulPartitionedCall:output:0lstm_21_39102801lstm_21_39102803lstm_21_39102805*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_21_layer_call_and_return_conditional_losses_391024722!
lstm_21/StatefulPartitionedCall┐
"dropout_21/StatefulPartitionedCallStatefulPartitionedCall(lstm_21/StatefulPartitionedCall:output:0#^dropout_20/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         `* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_21_layer_call_and_return_conditional_losses_391023052$
"dropout_21/StatefulPartitionedCall├
 dense_10/StatefulPartitionedCallStatefulPartitionedCall+dropout_21/StatefulPartitionedCall:output:0dense_10_39102809dense_10_39102811*
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
F__inference_dense_10_layer_call_and_return_conditional_losses_391022492"
 dense_10/StatefulPartitionedCallИ
IdentityIdentity)dense_10/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity 
NoOpNoOp!^dense_10/StatefulPartitionedCall#^dropout_20/StatefulPartitionedCall#^dropout_21/StatefulPartitionedCall ^lstm_20/StatefulPartitionedCall ^lstm_21/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2D
 dense_10/StatefulPartitionedCall dense_10/StatefulPartitionedCall2H
"dropout_20/StatefulPartitionedCall"dropout_20/StatefulPartitionedCall2H
"dropout_21/StatefulPartitionedCall"dropout_21/StatefulPartitionedCall2B
lstm_20/StatefulPartitionedCalllstm_20/StatefulPartitionedCall2B
lstm_21/StatefulPartitionedCalllstm_21/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_20_input"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┐
serving_defaultл
K
lstm_20_input:
serving_default_lstm_20_input:0         ]@
dense_104
StatefulPartitionedCall:0         tensorflow/serving/predict:∙╗
°
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
А_default_save_signature
+Б&call_and_return_all_conditional_losses
В__call__"
_tf_keras_sequential
┼
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+Г&call_and_return_all_conditional_losses
Д__call__"
_tf_keras_rnn_layer
з
regularization_losses
	variables
trainable_variables
	keras_api
+Е&call_and_return_all_conditional_losses
Ж__call__"
_tf_keras_layer
┼
cell

state_spec
regularization_losses
	variables
trainable_variables
	keras_api
+З&call_and_return_all_conditional_losses
И__call__"
_tf_keras_rnn_layer
з
regularization_losses
	variables
trainable_variables
	keras_api
+Й&call_and_return_all_conditional_losses
К__call__"
_tf_keras_layer
╜

 kernel
!bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
+Л&call_and_return_all_conditional_losses
М__call__"
_tf_keras_layer
у
&iter

'beta_1

(beta_2
	)decay
*learning_rate mp!mq+mr,ms-mt.mu/mv0mw vx!vy+vz,v{-v|.v}/v~0v"
	optimizer
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
╬
regularization_losses
	variables
1metrics
2layer_metrics
3layer_regularization_losses

4layers
5non_trainable_variables
	trainable_variables
В__call__
А_default_save_signature
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
-
Нserving_default"
signature_map
у
6
state_size

+kernel
,recurrent_kernel
-bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+О&call_and_return_all_conditional_losses
П__call__"
_tf_keras_layer
 "
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
╝
regularization_losses
	variables
;metrics

<states
=layer_metrics
>layer_regularization_losses

?layers
@non_trainable_variables
trainable_variables
Д__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
regularization_losses
Ametrics
	variables
Blayer_metrics
Clayer_regularization_losses
trainable_variables
Dnon_trainable_variables

Elayers
Ж__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
у
F
state_size

.kernel
/recurrent_kernel
0bias
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"
_tf_keras_layer
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
╝
regularization_losses
	variables
Kmetrics

Lstates
Mlayer_metrics
Nlayer_regularization_losses

Olayers
Pnon_trainable_variables
trainable_variables
И__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
regularization_losses
Qmetrics
	variables
Rlayer_metrics
Slayer_regularization_losses
trainable_variables
Tnon_trainable_variables

Ulayers
К__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
!:`2dense_10/kernel
:2dense_10/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
░
"regularization_losses
Vmetrics
#	variables
Wlayer_metrics
Xlayer_regularization_losses
$trainable_variables
Ynon_trainable_variables

Zlayers
М__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	]°2lstm_20/lstm_cell_20/kernel
9:7
Ю°2%lstm_20/lstm_cell_20/recurrent_kernel
(:&°2lstm_20/lstm_cell_20/bias
/:-
ЮА2lstm_21/lstm_cell_21/kernel
8:6	`А2%lstm_21/lstm_cell_21/recurrent_kernel
(:&А2lstm_21/lstm_cell_21/bias
.
[0
\1"
trackable_list_wrapper
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
 "
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
░
7regularization_losses
]metrics
8	variables
^layer_metrics
_layer_regularization_losses
9trainable_variables
`non_trainable_variables

alayers
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
░
Gregularization_losses
bmetrics
H	variables
clayer_metrics
dlayer_regularization_losses
Itrainable_variables
enon_trainable_variables

flayers
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
&:$`2Adam/dense_10/kernel/m
 :2Adam/dense_10/bias/m
3:1	]°2"Adam/lstm_20/lstm_cell_20/kernel/m
>:<
Ю°2,Adam/lstm_20/lstm_cell_20/recurrent_kernel/m
-:+°2 Adam/lstm_20/lstm_cell_20/bias/m
4:2
ЮА2"Adam/lstm_21/lstm_cell_21/kernel/m
=:;	`А2,Adam/lstm_21/lstm_cell_21/recurrent_kernel/m
-:+А2 Adam/lstm_21/lstm_cell_21/bias/m
&:$`2Adam/dense_10/kernel/v
 :2Adam/dense_10/bias/v
3:1	]°2"Adam/lstm_20/lstm_cell_20/kernel/v
>:<
Ю°2,Adam/lstm_20/lstm_cell_20/recurrent_kernel/v
-:+°2 Adam/lstm_20/lstm_cell_20/bias/v
4:2
ЮА2"Adam/lstm_21/lstm_cell_21/kernel/v
=:;	`А2,Adam/lstm_21/lstm_cell_21/recurrent_kernel/v
-:+А2 Adam/lstm_21/lstm_cell_21/bias/v
╘B╤
#__inference__wrapped_model_39100621lstm_20_input"Ш
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103171
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103512
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102790
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102815└
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
0__inference_sequential_10_layer_call_fn_39102275
0__inference_sequential_10_layer_call_fn_39103533
0__inference_sequential_10_layer_call_fn_39103554
0__inference_sequential_10_layer_call_fn_39102765└
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
E__inference_lstm_20_layer_call_and_return_conditional_losses_39103705
E__inference_lstm_20_layer_call_and_return_conditional_losses_39103856
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104007
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104158╒
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
*__inference_lstm_20_layer_call_fn_39104169
*__inference_lstm_20_layer_call_fn_39104180
*__inference_lstm_20_layer_call_fn_39104191
*__inference_lstm_20_layer_call_fn_39104202╒
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
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104207
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104219┤
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
-__inference_dropout_20_layer_call_fn_39104224
-__inference_dropout_20_layer_call_fn_39104229┤
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
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104380
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104531
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104682
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104833╒
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
*__inference_lstm_21_layer_call_fn_39104844
*__inference_lstm_21_layer_call_fn_39104855
*__inference_lstm_21_layer_call_fn_39104866
*__inference_lstm_21_layer_call_fn_39104877╒
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
H__inference_dropout_21_layer_call_and_return_conditional_losses_39104882
H__inference_dropout_21_layer_call_and_return_conditional_losses_39104894┤
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
-__inference_dropout_21_layer_call_fn_39104899
-__inference_dropout_21_layer_call_fn_39104904┤
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
F__inference_dense_10_layer_call_and_return_conditional_losses_39104935в
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
+__inference_dense_10_layer_call_fn_39104944в
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
&__inference_signature_wrapper_39102844lstm_20_input"Ф
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
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39104976
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105008╛
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
/__inference_lstm_cell_20_layer_call_fn_39105025
/__inference_lstm_cell_20_layer_call_fn_39105042╛
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
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105074
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105106╛
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
/__inference_lstm_cell_21_layer_call_fn_39105123
/__inference_lstm_cell_21_layer_call_fn_39105140╛
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
#__inference__wrapped_model_39100621+,-./0 !:в7
0в-
+К(
lstm_20_input         ]
к "7к4
2
dense_10&К#
dense_10         о
F__inference_dense_10_layer_call_and_return_conditional_losses_39104935d !3в0
)в&
$К!
inputs         `
к ")в&
К
0         
Ъ Ж
+__inference_dense_10_layer_call_fn_39104944W !3в0
)в&
$К!
inputs         `
к "К         ▓
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104207f8в5
.в+
%К"
inputs         Ю
p 
к "*в'
 К
0         Ю
Ъ ▓
H__inference_dropout_20_layer_call_and_return_conditional_losses_39104219f8в5
.в+
%К"
inputs         Ю
p
к "*в'
 К
0         Ю
Ъ К
-__inference_dropout_20_layer_call_fn_39104224Y8в5
.в+
%К"
inputs         Ю
p 
к "К         ЮК
-__inference_dropout_20_layer_call_fn_39104229Y8в5
.в+
%К"
inputs         Ю
p
к "К         Ю░
H__inference_dropout_21_layer_call_and_return_conditional_losses_39104882d7в4
-в*
$К!
inputs         `
p 
к ")в&
К
0         `
Ъ ░
H__inference_dropout_21_layer_call_and_return_conditional_losses_39104894d7в4
-в*
$К!
inputs         `
p
к ")в&
К
0         `
Ъ И
-__inference_dropout_21_layer_call_fn_39104899W7в4
-в*
$К!
inputs         `
p 
к "К         `И
-__inference_dropout_21_layer_call_fn_39104904W7в4
-в*
$К!
inputs         `
p
к "К         `╒
E__inference_lstm_20_layer_call_and_return_conditional_losses_39103705Л+,-OвL
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
0                  Ю
Ъ ╒
E__inference_lstm_20_layer_call_and_return_conditional_losses_39103856Л+,-OвL
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
0                  Ю
Ъ ╗
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104007r+,-?в<
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
0         Ю
Ъ ╗
E__inference_lstm_20_layer_call_and_return_conditional_losses_39104158r+,-?в<
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
0         Ю
Ъ м
*__inference_lstm_20_layer_call_fn_39104169~+,-OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "&К#                  Юм
*__inference_lstm_20_layer_call_fn_39104180~+,-OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "&К#                  ЮУ
*__inference_lstm_20_layer_call_fn_39104191e+,-?в<
5в2
$К!
inputs         ]

 
p 

 
к "К         ЮУ
*__inference_lstm_20_layer_call_fn_39104202e+,-?в<
5в2
$К!
inputs         ]

 
p

 
к "К         Ю╒
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104380Л./0PвM
FвC
5Ъ2
0К-
inputs/0                  Ю

 
p 

 
к "2в/
(К%
0                  `
Ъ ╒
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104531Л./0PвM
FвC
5Ъ2
0К-
inputs/0                  Ю

 
p

 
к "2в/
(К%
0                  `
Ъ ╗
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104682r./0@в=
6в3
%К"
inputs         Ю

 
p 

 
к ")в&
К
0         `
Ъ ╗
E__inference_lstm_21_layer_call_and_return_conditional_losses_39104833r./0@в=
6в3
%К"
inputs         Ю

 
p

 
к ")в&
К
0         `
Ъ м
*__inference_lstm_21_layer_call_fn_39104844~./0PвM
FвC
5Ъ2
0К-
inputs/0                  Ю

 
p 

 
к "%К"                  `м
*__inference_lstm_21_layer_call_fn_39104855~./0PвM
FвC
5Ъ2
0К-
inputs/0                  Ю

 
p

 
к "%К"                  `У
*__inference_lstm_21_layer_call_fn_39104866e./0@в=
6в3
%К"
inputs         Ю

 
p 

 
к "К         `У
*__inference_lstm_21_layer_call_fn_39104877e./0@в=
6в3
%К"
inputs         Ю

 
p

 
к "К         `╤
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39104976В+,-Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         Ю
#К 
states/1         Ю
p 
к "vвs
lвi
К
0/0         Ю
GЪD
 К
0/1/0         Ю
 К
0/1/1         Ю
Ъ ╤
J__inference_lstm_cell_20_layer_call_and_return_conditional_losses_39105008В+,-Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         Ю
#К 
states/1         Ю
p
к "vвs
lвi
К
0/0         Ю
GЪD
 К
0/1/0         Ю
 К
0/1/1         Ю
Ъ ж
/__inference_lstm_cell_20_layer_call_fn_39105025Є+,-Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         Ю
#К 
states/1         Ю
p 
к "fвc
К
0         Ю
CЪ@
К
1/0         Ю
К
1/1         Юж
/__inference_lstm_cell_20_layer_call_fn_39105042Є+,-Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         Ю
#К 
states/1         Ю
p
к "fвc
К
0         Ю
CЪ@
К
1/0         Ю
К
1/1         Ю═
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105074■./0Бв~
wвt
!К
inputs         Ю
KвH
"К
states/0         `
"К
states/1         `
p 
к "sвp
iвf
К
0/0         `
EЪB
К
0/1/0         `
К
0/1/1         `
Ъ ═
J__inference_lstm_cell_21_layer_call_and_return_conditional_losses_39105106■./0Бв~
wвt
!К
inputs         Ю
KвH
"К
states/0         `
"К
states/1         `
p
к "sвp
iвf
К
0/0         `
EЪB
К
0/1/0         `
К
0/1/1         `
Ъ в
/__inference_lstm_cell_21_layer_call_fn_39105123ю./0Бв~
wвt
!К
inputs         Ю
KвH
"К
states/0         `
"К
states/1         `
p 
к "cв`
К
0         `
AЪ>
К
1/0         `
К
1/1         `в
/__inference_lstm_cell_21_layer_call_fn_39105140ю./0Бв~
wвt
!К
inputs         Ю
KвH
"К
states/0         `
"К
states/1         `
p
к "cв`
К
0         `
AЪ>
К
1/0         `
К
1/1         `╚
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102790y+,-./0 !Bв?
8в5
+К(
lstm_20_input         ]
p 

 
к ")в&
К
0         
Ъ ╚
K__inference_sequential_10_layer_call_and_return_conditional_losses_39102815y+,-./0 !Bв?
8в5
+К(
lstm_20_input         ]
p

 
к ")в&
К
0         
Ъ ┴
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103171r+,-./0 !;в8
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
K__inference_sequential_10_layer_call_and_return_conditional_losses_39103512r+,-./0 !;в8
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
0__inference_sequential_10_layer_call_fn_39102275l+,-./0 !Bв?
8в5
+К(
lstm_20_input         ]
p 

 
к "К         а
0__inference_sequential_10_layer_call_fn_39102765l+,-./0 !Bв?
8в5
+К(
lstm_20_input         ]
p

 
к "К         Щ
0__inference_sequential_10_layer_call_fn_39103533e+,-./0 !;в8
1в.
$К!
inputs         ]
p 

 
к "К         Щ
0__inference_sequential_10_layer_call_fn_39103554e+,-./0 !;в8
1в.
$К!
inputs         ]
p

 
к "К         ╗
&__inference_signature_wrapper_39102844Р+,-./0 !KвH
в 
Aк>
<
lstm_20_input+К(
lstm_20_input         ]"7к4
2
dense_10&К#
dense_10         