╙э'
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
И"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ЭС&
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	у*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	у*
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
У
lstm_12/lstm_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]Ш*,
shared_namelstm_12/lstm_cell_12/kernel
М
/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/kernel*
_output_shapes
:	]Ш*
dtype0
и
%lstm_12/lstm_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
жШ*6
shared_name'%lstm_12/lstm_cell_12/recurrent_kernel
б
9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_12/lstm_cell_12/recurrent_kernel* 
_output_shapes
:
жШ*
dtype0
Л
lstm_12/lstm_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш**
shared_namelstm_12/lstm_cell_12/bias
Д
-lstm_12/lstm_cell_12/bias/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/bias*
_output_shapes	
:Ш*
dtype0
Ф
lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
жМ*,
shared_namelstm_13/lstm_cell_13/kernel
Н
/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/kernel* 
_output_shapes
:
жМ*
dtype0
и
%lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
уМ*6
shared_name'%lstm_13/lstm_cell_13/recurrent_kernel
б
9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_13/lstm_cell_13/recurrent_kernel* 
_output_shapes
:
уМ*
dtype0
Л
lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:М**
shared_namelstm_13/lstm_cell_13/bias
Д
-lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/bias*
_output_shapes	
:М*
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
З
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	у*&
shared_nameAdam/dense_6/kernel/m
А
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	у*
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
б
"Adam/lstm_12/lstm_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]Ш*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/m
Ъ
6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/m*
_output_shapes
:	]Ш*
dtype0
╢
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
жШ*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
п
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m* 
_output_shapes
:
жШ*
dtype0
Щ
 Adam/lstm_12/lstm_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/m
Т
4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/m*
_output_shapes	
:Ш*
dtype0
в
"Adam/lstm_13/lstm_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
жМ*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/m
Ы
6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/m* 
_output_shapes
:
жМ*
dtype0
╢
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
уМ*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
п
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m* 
_output_shapes
:
уМ*
dtype0
Щ
 Adam/lstm_13/lstm_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:М*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/m
Т
4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/m*
_output_shapes	
:М*
dtype0
З
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	у*&
shared_nameAdam/dense_6/kernel/v
А
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	у*
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
б
"Adam/lstm_12/lstm_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]Ш*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/v
Ъ
6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/v*
_output_shapes
:	]Ш*
dtype0
╢
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
жШ*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
п
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v* 
_output_shapes
:
жШ*
dtype0
Щ
 Adam/lstm_12/lstm_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ш*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/v
Т
4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/v*
_output_shapes	
:Ш*
dtype0
в
"Adam/lstm_13/lstm_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
жМ*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/v
Ы
6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/v* 
_output_shapes
:
жМ*
dtype0
╢
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
уМ*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
п
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v* 
_output_shapes
:
уМ*
dtype0
Щ
 Adam/lstm_13/lstm_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:М*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/v
Т
4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/v*
_output_shapes	
:М*
dtype0

NoOpNoOp
°7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*│7
valueй7Bж7 BЯ7
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
╨
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
н
1non_trainable_variables
2metrics

3layers
4layer_metrics
trainable_variables
regularization_losses
		variables
5layer_regularization_losses
 
О
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
╣
;non_trainable_variables
<metrics

=states

>layers
?layer_metrics
trainable_variables
regularization_losses
	variables
@layer_regularization_losses
 
 
 
н
Anon_trainable_variables
Bmetrics

Clayers
Dlayer_metrics
trainable_variables
regularization_losses
	variables
Elayer_regularization_losses
О
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
╣
Knon_trainable_variables
Lmetrics

Mstates

Nlayers
Olayer_metrics
trainable_variables
regularization_losses
	variables
Player_regularization_losses
 
 
 
н
Qnon_trainable_variables
Rmetrics

Slayers
Tlayer_metrics
trainable_variables
regularization_losses
	variables
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
н
Vnon_trainable_variables
Wmetrics

Xlayers
Ylayer_metrics
"trainable_variables
#regularization_losses
$	variables
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
 

[0
\1
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
 

+0
,1
-2
н
]non_trainable_variables
^metrics

_layers
`layer_metrics
7trainable_variables
8regularization_losses
9	variables
alayer_regularization_losses
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
 

.0
/1
02
н
bnon_trainable_variables
cmetrics

dlayers
elayer_metrics
Gtrainable_variables
Hregularization_losses
I	variables
flayer_regularization_losses
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
ЕВ
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ЕВ
VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ПМ
VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
И
serving_default_lstm_12_inputPlaceholder*+
_output_shapes
:         ]*
dtype0* 
shape:         ]
к
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_12_inputlstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biaslstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biasdense_6/kerneldense_6/bias*
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
&__inference_signature_wrapper_26063088
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╗
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
GPU 2J 8В **
f%R#
!__inference__traced_save_26065506
в	
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_26065615аЇ$
╘!
¤
E__inference_dense_6_layer_call_and_return_conditional_losses_26065188

inputs4
!tensordot_readvariableop_resource:	у-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	у*
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
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         у2
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
_construction_contextkEagerRuntime*/
_input_shapes
:         у: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
Й
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_26065136

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         у2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         у2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         у:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
░?
╘
while_body_26064211
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]ШI
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]ШG
3while_lstm_cell_12_matmul_1_readvariableop_resource:
жШA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв)while/lstm_cell_12/BiasAdd/ReadVariableOpв(while/lstm_cell_12/MatMul/ReadVariableOpв*while/lstm_cell_12/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp╫
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul╨
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp└
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul_1╕
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/add╚
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp┼
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/BiasAddК
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dimП
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
while/lstm_cell_12/splitЩ
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/SigmoidЭ
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_1б
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mulР
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu╡
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_1к
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/add_1Э
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_2П
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu_1╣
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2V
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_26064734
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064734___redundant_placeholder06
2while_while_cond_26064734___redundant_placeholder16
2while_while_cond_26064734___redundant_placeholder26
2while_while_cond_26064734___redundant_placeholder3
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
К\
Я
E__inference_lstm_13_layer_call_and_return_conditional_losses_26062716

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
жМA
-lstm_cell_13_matmul_1_readvariableop_resource:
уМ;
,lstm_cell_13_biasadd_readvariableop_resource:	М
identityИв#lstm_cell_13/BiasAdd/ReadVariableOpв"lstm_cell_13/MatMul/ReadVariableOpв$lstm_cell_13/MatMul_1/ReadVariableOpвwhileD
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
B :у2
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
B :у2
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
:         у2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
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
B :у2
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
:         у2	
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
:         ж2
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
valueB"    ж  27
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
:         ж*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOpн
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul╝
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOpй
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul_1а
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/add┤
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOpн
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimў
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
lstm_cell_13/splitЗ
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/SigmoidЛ
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_1М
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
lstm_cell_13/ReluЭ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_1Т
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/add_1Л
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/Relu_1б
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26062632*
condR
while_cond_26062631*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         у*
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
:         у*
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
:         у2
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
:         у2

Identity╚
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ж: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
Ў°
З
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063457

inputsF
3lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]ШI
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
жШC
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	ШG
3lstm_13_lstm_cell_13_matmul_readvariableop_resource:
жМI
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
уМC
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	М<
)dense_6_tensordot_readvariableop_resource:	у5
'dense_6_biasadd_readvariableop_resource:
identityИвdense_6/BiasAdd/ReadVariableOpв dense_6/Tensordot/ReadVariableOpв+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpв*lstm_12/lstm_cell_12/MatMul/ReadVariableOpв,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpвlstm_12/whileв+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpв*lstm_13/lstm_cell_13/MatMul/ReadVariableOpв,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpвlstm_13/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/ShapeД
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stackИ
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1И
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2Т
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
B :ж2
lstm_12/zeros/mul/yМ
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
B :ш2
lstm_12/zeros/Less/yЗ
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
B :ж2
lstm_12/zeros/packed/1г
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
lstm_12/zeros/ConstЦ
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:         ж2
lstm_12/zerosq
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
lstm_12/zeros_1/mul/yТ
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
B :ш2
lstm_12/zeros_1/Less/yП
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
B :ж2
lstm_12/zeros_1/packed/1й
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
lstm_12/zeros_1/ConstЮ
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ж2
lstm_12/zeros_1Е
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/permТ
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1И
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stackМ
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1М
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2Ю
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1Х
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_12/TensorArrayV2/element_shape╥
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2╧
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensorИ
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stackМ
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1М
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2м
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_12/strided_slice_2═
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp═
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_12/lstm_cell_12/MatMul╘
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp╔
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_12/lstm_cell_12/MatMul_1└
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
lstm_12/lstm_cell_12/add╠
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp═
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_12/lstm_cell_12/BiasAddО
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dimЧ
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
lstm_12/lstm_cell_12/splitЯ
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/Sigmoidг
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2 
lstm_12/lstm_cell_12/Sigmoid_1м
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/mulЦ
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/Relu╜
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/mul_1▓
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/add_1г
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2 
lstm_12/lstm_cell_12/Sigmoid_2Х
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/Relu_1┴
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/mul_2Я
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2'
%lstm_12/TensorArrayV2_1/element_shape╪
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
lstm_12/timeП
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counterЛ
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_12_while_body_26063197*'
condR
lstm_12_while_cond_26063196*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
lstm_12/while┼
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ж*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStackС
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_12/strided_slice_3/stackМ
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1М
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2╦
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ж*
shrink_axis_mask2
lstm_12/strided_slice_3Й
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/perm╞
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ж2
lstm_12/transpose_1v
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/runtimeЖ
dropout_12/IdentityIdentitylstm_12/transpose_1:y:0*
T0*,
_output_shapes
:         ж2
dropout_12/Identityj
lstm_13/ShapeShapedropout_12/Identity:output:0*
T0*
_output_shapes
:2
lstm_13/ShapeД
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stackИ
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1И
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2Т
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
B :у2
lstm_13/zeros/mul/yМ
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
B :ш2
lstm_13/zeros/Less/yЗ
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
B :у2
lstm_13/zeros/packed/1г
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
lstm_13/zeros/ConstЦ
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:         у2
lstm_13/zerosq
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
lstm_13/zeros_1/mul/yТ
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
B :ш2
lstm_13/zeros_1/Less/yП
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
B :у2
lstm_13/zeros_1/packed/1й
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
lstm_13/zeros_1/ConstЮ
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:         у2
lstm_13/zeros_1Е
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/permй
lstm_13/transpose	Transposedropout_12/Identity:output:0lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:         ж2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1И
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stackМ
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1М
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2Ю
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1Х
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_13/TensorArrayV2/element_shape╥
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2╧
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensorИ
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stackМ
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1М
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2н
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ж*
shrink_axis_mask2
lstm_13/strided_slice_2╬
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp═
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_13/lstm_cell_13/MatMul╘
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp╔
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_13/lstm_cell_13/MatMul_1└
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
lstm_13/lstm_cell_13/add╠
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp═
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_13/lstm_cell_13/BiasAddО
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dimЧ
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
lstm_13/lstm_cell_13/splitЯ
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/Sigmoidг
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2 
lstm_13/lstm_cell_13/Sigmoid_1м
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/mulЦ
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/Relu╜
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/mul_1▓
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/add_1г
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2 
lstm_13/lstm_cell_13/Sigmoid_2Х
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/Relu_1┴
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/mul_2Я
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2'
%lstm_13/TensorArrayV2_1/element_shape╪
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
lstm_13/timeП
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counterЛ
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_13_while_body_26063345*'
condR
lstm_13_while_cond_26063344*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
lstm_13/while┼
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         у*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStackС
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_13/strided_slice_3/stackМ
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1М
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2╦
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         у*
shrink_axis_mask2
lstm_13/strided_slice_3Й
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/perm╞
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:         у2
lstm_13/transpose_1v
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/runtimeЖ
dropout_13/IdentityIdentitylstm_13/transpose_1:y:0*
T0*,
_output_shapes
:         у2
dropout_13/Identityп
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	у*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axesБ
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
dense_6/Tensordot/ShapeД
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis∙
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2И
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis 
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
dense_6/Tensordot/Constа
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/ProdА
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1и
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1А
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis╪
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concatм
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack┐
dense_6/Tensordot/transpose	Transposedropout_13/Identity:output:0!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:         у2
dense_6/Tensordot/transpose┐
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_6/Tensordot/Reshape╛
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/Tensordot/MatMulА
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/Const_2Д
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axisх
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1░
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_6/Tensordotд
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOpз
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_6/BiasAdd}
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_6/Softmaxx
IdentityIdentitydense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╞
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2@
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
:         ]
 
_user_specified_nameinputs
╜
╛
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063059
lstm_12_input#
lstm_12_26063037:	]Ш$
lstm_12_26063039:
жШ
lstm_12_26063041:	Ш$
lstm_13_26063045:
жМ$
lstm_13_26063047:
уМ
lstm_13_26063049:	М#
dense_6_26063053:	у
dense_6_26063055:
identityИвdense_6/StatefulPartitionedCallв"dropout_12/StatefulPartitionedCallв"dropout_13/StatefulPartitionedCallвlstm_12/StatefulPartitionedCallвlstm_13/StatefulPartitionedCall╡
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_26063037lstm_12_26063039lstm_12_26063041*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260629122!
lstm_12/StatefulPartitionedCallЫ
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260627452$
"dropout_12/StatefulPartitionedCall╙
lstm_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0lstm_13_26063045lstm_13_26063047lstm_13_26063049*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260627162!
lstm_13/StatefulPartitionedCall└
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260625492$
"dropout_13/StatefulPartitionedCall╛
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_6_26063053dense_6_26063055*
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
GPU 2J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_260624932!
dense_6/StatefulPartitionedCallЗ
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity■
NoOpNoOp ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_12_input
╘

э
lstm_12_while_cond_26063196,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1F
Blstm_12_while_lstm_12_while_cond_26063196___redundant_placeholder0F
Blstm_12_while_lstm_12_while_cond_26063196___redundant_placeholder1F
Blstm_12_while_lstm_12_while_cond_26063196___redundant_placeholder2F
Blstm_12_while_lstm_12_while_cond_26063196___redundant_placeholder3
lstm_12_while_identity
Ш
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
Ч
К
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26065384

inputs
states_0
states_12
matmul_readvariableop_resource:
жМ4
 matmul_1_readvariableop_resource:
уМ.
biasadd_readvariableop_resource:	М
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         М2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2	
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
P:         у:         у:         у:         у*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         у2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         у2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         у2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         у2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         у2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         у2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         у2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         у2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         у2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identity_2Щ
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
B:         ж:         у:         у: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:RN
(
_output_shapes
:         у
"
_user_specified_name
states/0:RN
(
_output_shapes
:         у
"
_user_specified_name
states/1
Д\
Ю
E__inference_lstm_12_layer_call_and_return_conditional_losses_26062282

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]ШA
-lstm_cell_12_matmul_1_readvariableop_resource:
жШ;
,lstm_cell_12_biasadd_readvariableop_resource:	Ш
identityИв#lstm_cell_12/BiasAdd/ReadVariableOpв"lstm_cell_12/MatMul/ReadVariableOpв$lstm_cell_12/MatMul_1/ReadVariableOpвwhileD
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
B :ж2
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
B :ж2
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
:         ж2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
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
B :ж2
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
:         ж2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOpн
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul╝
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOpй
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul_1а
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/add┤
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOpн
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimў
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
lstm_cell_12/splitЗ
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/SigmoidЛ
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_1М
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
lstm_cell_12/ReluЭ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_1Т
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/add_1Л
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Relu_1б
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26062198*
condR
while_cond_26062197*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ж*
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
:         ж*
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
:         ж2
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
:         ж2

Identity╚
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
щJ
╓

lstm_13_while_body_26063679,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
жМQ
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМK
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	М
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorM
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
жМO
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
уМI
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	МИв1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpв0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpв2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp╙
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItemт
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpў
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2#
!lstm_13/while/lstm_cell_13/MatMulш
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpр
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2%
#lstm_13/while/lstm_cell_13/MatMul_1╪
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2 
lstm_13/while/lstm_cell_13/addр
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpх
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2$
"lstm_13/while/lstm_cell_13/BiasAddЪ
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dimп
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2"
 lstm_13/while/lstm_cell_13/split▒
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2$
"lstm_13/while/lstm_cell_13/Sigmoid╡
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2&
$lstm_13/while/lstm_cell_13/Sigmoid_1┴
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:         у2 
lstm_13/while/lstm_cell_13/mulи
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2!
lstm_13/while/lstm_cell_13/Relu╒
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2"
 lstm_13/while/lstm_cell_13/mul_1╩
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2"
 lstm_13/while/lstm_cell_13/add_1╡
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2&
$lstm_13/while/lstm_cell_13/Sigmoid_2з
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2#
!lstm_13/while/lstm_cell_13/Relu_1┘
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2"
 lstm_13/while/lstm_cell_13/mul_2И
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
lstm_13/while/add/yЙ
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
lstm_13/while/add_1/yЮ
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1Л
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identityж
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1Н
lstm_13/while/Identity_2Identitylstm_13/while/add:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2║
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3о
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:         у2
lstm_13/while/Identity_4о
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:         у2
lstm_13/while/Identity_5Ж
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
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"╚
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2f
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
├\
а
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064144
inputs_0>
+lstm_cell_12_matmul_readvariableop_resource:	]ШA
-lstm_cell_12_matmul_1_readvariableop_resource:
жШ;
,lstm_cell_12_biasadd_readvariableop_resource:	Ш
identityИв#lstm_cell_12/BiasAdd/ReadVariableOpв"lstm_cell_12/MatMul/ReadVariableOpв$lstm_cell_12/MatMul_1/ReadVariableOpвwhileF
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
B :ж2
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
B :ж2
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
:         ж2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
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
B :ж2
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
:         ж2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOpн
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul╝
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOpй
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul_1а
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/add┤
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOpн
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimў
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
lstm_cell_12/splitЗ
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/SigmoidЛ
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_1М
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
lstm_cell_12/ReluЭ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_1Т
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/add_1Л
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Relu_1б
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064060*
condR
while_cond_26064059*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ж*
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
:         ж*
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
!:                  ж2
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
!:                  ж2

Identity╚
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
Й
f
H__inference_dropout_12_layer_call_and_return_conditional_losses_26064461

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ж2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ж2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ж:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
├\
а
E__inference_lstm_12_layer_call_and_return_conditional_losses_26063993
inputs_0>
+lstm_cell_12_matmul_readvariableop_resource:	]ШA
-lstm_cell_12_matmul_1_readvariableop_resource:
жШ;
,lstm_cell_12_biasadd_readvariableop_resource:	Ш
identityИв#lstm_cell_12/BiasAdd/ReadVariableOpв"lstm_cell_12/MatMul/ReadVariableOpв$lstm_cell_12/MatMul_1/ReadVariableOpвwhileF
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
B :ж2
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
B :ж2
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
:         ж2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
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
B :ж2
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
:         ж2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOpн
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul╝
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOpй
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul_1а
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/add┤
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOpн
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimў
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
lstm_cell_12/splitЗ
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/SigmoidЛ
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_1М
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
lstm_cell_12/ReluЭ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_1Т
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/add_1Л
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Relu_1б
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26063909*
condR
while_cond_26063908*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ж*
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
:         ж*
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
!:                  ж2
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
!:                  ж2

Identity╚
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  ]
"
_user_specified_name
inputs/0
П
И
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26061570

inputs

states
states_12
matmul_readvariableop_resource:
жМ4
 matmul_1_readvariableop_resource:
уМ.
biasadd_readvariableop_resource:	М
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         М2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2	
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
P:         у:         у:         у:         у*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         у2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         у2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         у2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         у2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         у2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         у2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         у2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         у2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         у2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identity_2Щ
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
B:         ж:         у:         у: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:PL
(
_output_shapes
:         у
 
_user_specified_namestates:PL
(
_output_shapes
:         у
 
_user_specified_namestates
╔\
б
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064668
inputs_0?
+lstm_cell_13_matmul_readvariableop_resource:
жМA
-lstm_cell_13_matmul_1_readvariableop_resource:
уМ;
,lstm_cell_13_biasadd_readvariableop_resource:	М
identityИв#lstm_cell_13/BiasAdd/ReadVariableOpв"lstm_cell_13/MatMul/ReadVariableOpв$lstm_cell_13/MatMul_1/ReadVariableOpвwhileF
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
B :у2
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
B :у2
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
:         у2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
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
B :у2
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
:         у2	
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
!:                  ж2
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
valueB"    ж  27
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
:         ж*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOpн
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul╝
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOpй
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul_1а
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/add┤
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOpн
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimў
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
lstm_cell_13/splitЗ
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/SigmoidЛ
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_1М
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
lstm_cell_13/ReluЭ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_1Т
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/add_1Л
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/Relu_1б
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064584*
condR
while_cond_26064583*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  у*
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
:         у*
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
!:                  у2
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
!:                  у2

Identity╚
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ж: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ж
"
_user_specified_name
inputs/0
╣
╣
*__inference_lstm_13_layer_call_fn_26064506

inputs
unknown:
жМ
	unknown_0:
уМ
	unknown_1:	М
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260624472
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         у2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ж: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
░?
╘
while_body_26063909
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]ШI
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]ШG
3while_lstm_cell_12_matmul_1_readvariableop_resource:
жШA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв)while/lstm_cell_12/BiasAdd/ReadVariableOpв(while/lstm_cell_12/MatMul/ReadVariableOpв*while/lstm_cell_12/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp╫
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul╨
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp└
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul_1╕
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/add╚
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp┼
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/BiasAddК
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dimП
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
while/lstm_cell_12/splitЩ
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/SigmoidЭ
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_1б
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mulР
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu╡
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_1к
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/add_1Э
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_2П
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu_1╣
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2V
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
у
═
while_cond_26064059
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064059___redundant_placeholder06
2while_while_cond_26064059___redundant_placeholder16
2while_while_cond_26064059___redundant_placeholder26
2while_while_cond_26064059___redundant_placeholder3
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
и
╖
J__inference_sequential_6_layer_call_and_return_conditional_losses_26062969

inputs#
lstm_12_26062947:	]Ш$
lstm_12_26062949:
жШ
lstm_12_26062951:	Ш$
lstm_13_26062955:
жМ$
lstm_13_26062957:
уМ
lstm_13_26062959:	М#
dense_6_26062963:	у
dense_6_26062965:
identityИвdense_6/StatefulPartitionedCallв"dropout_12/StatefulPartitionedCallв"dropout_13/StatefulPartitionedCallвlstm_12/StatefulPartitionedCallвlstm_13/StatefulPartitionedCallо
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_26062947lstm_12_26062949lstm_12_26062951*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260629122!
lstm_12/StatefulPartitionedCallЫ
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260627452$
"dropout_12/StatefulPartitionedCall╙
lstm_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0lstm_13_26062955lstm_13_26062957lstm_13_26062959*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260627162!
lstm_13/StatefulPartitionedCall└
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260625492$
"dropout_13/StatefulPartitionedCall╛
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_6_26062963dense_6_26062965*
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
GPU 2J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_260624932!
dense_6/StatefulPartitionedCallЗ
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity■
NoOpNoOp ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
Л
З
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26060940

inputs

states
states_11
matmul_readvariableop_resource:	]Ш4
 matmul_1_readvariableop_resource:
жШ.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
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
P:         ж:         ж:         ж:         ж*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ж2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ж2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ж2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ж2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ж2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ж2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ж2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ж2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ж2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ж2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ж2

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
A:         ]:         ж:         ж: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ж
 
_user_specified_namestates:PL
(
_output_shapes
:         ж
 
_user_specified_namestates
у
═
while_cond_26062362
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26062362___redundant_placeholder06
2while_while_cond_26062362___redundant_placeholder16
2while_while_cond_26062362___redundant_placeholder26
2while_while_cond_26062362___redundant_placeholder3
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
И&
ї
while_body_26061584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_13_26061608_0:
жМ1
while_lstm_cell_13_26061610_0:
уМ,
while_lstm_cell_13_26061612_0:	М
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_13_26061608:
жМ/
while_lstm_cell_13_26061610:
уМ*
while_lstm_cell_13_26061612:	МИв*while/lstm_cell_13/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_26061608_0while_lstm_cell_13_26061610_0while_lstm_cell_13_26061612_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         у:         у:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260615702,
*while/lstm_cell_13/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_5З

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
while_lstm_cell_13_26061608while_lstm_cell_13_26061608_0"<
while_lstm_cell_13_26061610while_lstm_cell_13_26061610_0"<
while_lstm_cell_13_26061612while_lstm_cell_13_26061612_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2X
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
р
║
*__inference_lstm_12_layer_call_fn_26063820
inputs_0
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260612332
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ж2

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
╦F
О
E__inference_lstm_12_layer_call_and_return_conditional_losses_26061233

inputs(
lstm_cell_12_26061151:	]Ш)
lstm_cell_12_26061153:
жШ$
lstm_cell_12_26061155:	Ш
identityИв$lstm_cell_12/StatefulPartitionedCallвwhileD
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
B :ж2
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
B :ж2
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
:         ж2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
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
B :ж2
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
:         ж2	
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
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_26061151lstm_cell_12_26061153lstm_cell_12_26061155*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ж:         ж:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260610862&
$lstm_cell_12/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_26061151lstm_cell_12_26061153lstm_cell_12_26061155*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26061164*
condR
while_cond_26061163*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ж*
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
:         ж*
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
!:                  ж2
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
!:                  ж2

Identity}
NoOpNoOp%^lstm_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
╫
g
H__inference_dropout_12_layer_call_and_return_conditional_losses_26064473

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
:         ж2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ж*
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
:         ж2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ж2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ж2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ж:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
у
═
while_cond_26061793
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26061793___redundant_placeholder06
2while_while_cond_26061793___redundant_placeholder16
2while_while_cond_26061793___redundant_placeholder26
2while_while_cond_26061793___redundant_placeholder3
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
у
╗
*__inference_lstm_13_layer_call_fn_26064484
inputs_0
unknown:
жМ
	unknown_0:
уМ
	unknown_1:	М
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260616532
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  у2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ж: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ж
"
_user_specified_name
inputs/0
╗
f
-__inference_dropout_13_layer_call_fn_26065131

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
:         у* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260625492
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         у2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         у22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
╪
I
-__inference_dropout_13_layer_call_fn_26065126

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
:         у* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260624602
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         у2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         у:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
╔\
б
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064819
inputs_0?
+lstm_cell_13_matmul_readvariableop_resource:
жМA
-lstm_cell_13_matmul_1_readvariableop_resource:
уМ;
,lstm_cell_13_biasadd_readvariableop_resource:	М
identityИв#lstm_cell_13/BiasAdd/ReadVariableOpв"lstm_cell_13/MatMul/ReadVariableOpв$lstm_cell_13/MatMul_1/ReadVariableOpвwhileF
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
B :у2
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
B :у2
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
:         у2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
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
B :у2
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
:         у2	
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
!:                  ж2
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
valueB"    ж  27
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
:         ж*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOpн
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul╝
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOpй
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul_1а
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/add┤
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOpн
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimў
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
lstm_cell_13/splitЗ
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/SigmoidЛ
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_1М
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
lstm_cell_13/ReluЭ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_1Т
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/add_1Л
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/Relu_1б
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064735*
condR
while_cond_26064734*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  у*
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
:         у*
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
!:                  у2
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
!:                  у2

Identity╚
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ж: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:_ [
5
_output_shapes#
!:                  ж
"
_user_specified_name
inputs/0
П
И
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26061716

inputs

states
states_12
matmul_readvariableop_resource:
жМ4
 matmul_1_readvariableop_resource:
уМ.
biasadd_readvariableop_resource:	М
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         М2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2	
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
P:         у:         у:         у:         у*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         у2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         у2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         у2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         у2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         у2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         у2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         у2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         у2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         у2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identity_2Щ
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
B:         ж:         у:         у: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:PL
(
_output_shapes
:         у
 
_user_specified_namestates:PL
(
_output_shapes
:         у
 
_user_specified_namestates
м]
ї
(sequential_6_lstm_12_while_body_26060605F
Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counterL
Hsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations*
&sequential_6_lstm_12_while_placeholder,
(sequential_6_lstm_12_while_placeholder_1,
(sequential_6_lstm_12_while_placeholder_2,
(sequential_6_lstm_12_while_placeholder_3E
Asequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1_0Б
}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]Ш^
Jsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШX
Isequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш'
#sequential_6_lstm_12_while_identity)
%sequential_6_lstm_12_while_identity_1)
%sequential_6_lstm_12_while_identity_2)
%sequential_6_lstm_12_while_identity_3)
%sequential_6_lstm_12_while_identity_4)
%sequential_6_lstm_12_while_identity_5C
?sequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1
{sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensorY
Fsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]Ш\
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
жШV
Gsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpв=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpв?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpэ
Lsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2N
Lsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape╤
>sequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0&sequential_6_lstm_12_while_placeholderUsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype02@
>sequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItemИ
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOpHsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype02?
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpл
.sequential_6/lstm_12/while/lstm_cell_12/MatMulMatMulEsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш20
.sequential_6/lstm_12/while/lstm_cell_12/MatMulП
?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpJsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype02A
?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpФ
0sequential_6/lstm_12/while/lstm_cell_12/MatMul_1MatMul(sequential_6_lstm_12_while_placeholder_2Gsequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш22
0sequential_6/lstm_12/while/lstm_cell_12/MatMul_1М
+sequential_6/lstm_12/while/lstm_cell_12/addAddV28sequential_6/lstm_12/while/lstm_cell_12/MatMul:product:0:sequential_6/lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2-
+sequential_6/lstm_12/while/lstm_cell_12/addЗ
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpIsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02@
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpЩ
/sequential_6/lstm_12/while/lstm_cell_12/BiasAddBiasAdd/sequential_6/lstm_12/while/lstm_cell_12/add:z:0Fsequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш21
/sequential_6/lstm_12/while/lstm_cell_12/BiasAdd┤
7sequential_6/lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_6/lstm_12/while/lstm_cell_12/split/split_dimу
-sequential_6/lstm_12/while/lstm_cell_12/splitSplit@sequential_6/lstm_12/while/lstm_cell_12/split/split_dim:output:08sequential_6/lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2/
-sequential_6/lstm_12/while/lstm_cell_12/split╪
/sequential_6/lstm_12/while/lstm_cell_12/SigmoidSigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж21
/sequential_6/lstm_12/while/lstm_cell_12/Sigmoid▄
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж23
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1ї
+sequential_6/lstm_12/while/lstm_cell_12/mulMul5sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1:y:0(sequential_6_lstm_12_while_placeholder_3*
T0*(
_output_shapes
:         ж2-
+sequential_6/lstm_12/while/lstm_cell_12/mul╧
,sequential_6/lstm_12/while/lstm_cell_12/ReluRelu6sequential_6/lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2.
,sequential_6/lstm_12/while/lstm_cell_12/ReluЙ
-sequential_6/lstm_12/while/lstm_cell_12/mul_1Mul3sequential_6/lstm_12/while/lstm_cell_12/Sigmoid:y:0:sequential_6/lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2/
-sequential_6/lstm_12/while/lstm_cell_12/mul_1■
-sequential_6/lstm_12/while/lstm_cell_12/add_1AddV2/sequential_6/lstm_12/while/lstm_cell_12/mul:z:01sequential_6/lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2/
-sequential_6/lstm_12/while/lstm_cell_12/add_1▄
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж23
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2╬
.sequential_6/lstm_12/while/lstm_cell_12/Relu_1Relu1sequential_6/lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж20
.sequential_6/lstm_12/while/lstm_cell_12/Relu_1Н
-sequential_6/lstm_12/while/lstm_cell_12/mul_2Mul5sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2:y:0<sequential_6/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2/
-sequential_6/lstm_12/while/lstm_cell_12/mul_2╔
?sequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_6_lstm_12_while_placeholder_1&sequential_6_lstm_12_while_placeholder1sequential_6/lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_6/lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_6/lstm_12/while/add/y╜
sequential_6/lstm_12/while/addAddV2&sequential_6_lstm_12_while_placeholder)sequential_6/lstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_12/while/addК
"sequential_6/lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_6/lstm_12/while/add_1/y▀
 sequential_6/lstm_12/while/add_1AddV2Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counter+sequential_6/lstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_12/while/add_1┐
#sequential_6/lstm_12/while/IdentityIdentity$sequential_6/lstm_12/while/add_1:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_6/lstm_12/while/Identityч
%sequential_6/lstm_12/while/Identity_1IdentityHsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_1┴
%sequential_6/lstm_12/while/Identity_2Identity"sequential_6/lstm_12/while/add:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_2ю
%sequential_6/lstm_12/while/Identity_3IdentityOsequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_3т
%sequential_6/lstm_12/while/Identity_4Identity1sequential_6/lstm_12/while/lstm_cell_12/mul_2:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*(
_output_shapes
:         ж2'
%sequential_6/lstm_12/while/Identity_4т
%sequential_6/lstm_12/while/Identity_5Identity1sequential_6/lstm_12/while/lstm_cell_12/add_1:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*(
_output_shapes
:         ж2'
%sequential_6/lstm_12/while/Identity_5╟
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
%sequential_6_lstm_12_while_identity_5.sequential_6/lstm_12/while/Identity_5:output:0"Ф
Gsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resourceIsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"Ц
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resourceJsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"Т
Fsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resourceHsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"Д
?sequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1Asequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1_0"№
{sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2А
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2~
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2В
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
хJ
╘

lstm_12_while_body_26063524,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]ШQ
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШK
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorL
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]ШO
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
жШI
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpв0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpв2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp╙
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItemс
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpў
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2#
!lstm_12/while/lstm_cell_12/MatMulш
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpр
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2%
#lstm_12/while/lstm_cell_12/MatMul_1╪
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2 
lstm_12/while/lstm_cell_12/addр
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpх
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2$
"lstm_12/while/lstm_cell_12/BiasAddЪ
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dimп
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2"
 lstm_12/while/lstm_cell_12/split▒
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2$
"lstm_12/while/lstm_cell_12/Sigmoid╡
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2&
$lstm_12/while/lstm_cell_12/Sigmoid_1┴
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*(
_output_shapes
:         ж2 
lstm_12/while/lstm_cell_12/mulи
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2!
lstm_12/while/lstm_cell_12/Relu╒
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2"
 lstm_12/while/lstm_cell_12/mul_1╩
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2"
 lstm_12/while/lstm_cell_12/add_1╡
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2&
$lstm_12/while/lstm_cell_12/Sigmoid_2з
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2#
!lstm_12/while/lstm_cell_12/Relu_1┘
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2"
 lstm_12/while/lstm_cell_12/mul_2И
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
lstm_12/while/add/yЙ
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
lstm_12/while/add_1/yЮ
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1Л
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identityж
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1Н
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2║
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3о
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:         ж2
lstm_12/while/Identity_4о
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:         ж2
lstm_12/while/Identity_5Ж
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
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"╚
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2f
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
Е&
є
while_body_26061164
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_12_26061188_0:	]Ш1
while_lstm_cell_12_26061190_0:
жШ,
while_lstm_cell_12_26061192_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_12_26061188:	]Ш/
while_lstm_cell_12_26061190:
жШ*
while_lstm_cell_12_26061192:	ШИв*while/lstm_cell_12/StatefulPartitionedCall├
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
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_26061188_0while_lstm_cell_12_26061190_0while_lstm_cell_12_26061192_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ж:         ж:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260610862,
*while/lstm_cell_12/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_5З

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
while_lstm_cell_12_26061188while_lstm_cell_12_26061188_0"<
while_lstm_cell_12_26061190while_lstm_cell_12_26061190_0"<
while_lstm_cell_12_26061192while_lstm_cell_12_26061192_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2X
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
┤?
╓
while_body_26065037
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
жМI
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	М
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
жМG
3while_lstm_cell_13_matmul_1_readvariableop_resource:
уМA
2while_lstm_cell_13_biasadd_readvariableop_resource:	МИв)while/lstm_cell_13/BiasAdd/ReadVariableOpв(while/lstm_cell_13/MatMul/ReadVariableOpв*while/lstm_cell_13/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp╫
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul╨
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp└
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul_1╕
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/add╚
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp┼
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/BiasAddК
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dimП
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
while/lstm_cell_13/splitЩ
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/SigmoidЭ
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_1б
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mulР
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu╡
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_1к
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/add_1Э
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_2П
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu_1╣
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2V
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
Й
f
H__inference_dropout_12_layer_call_and_return_conditional_losses_26062295

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         ж2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         ж2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ж:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
Ч

╠
/__inference_sequential_6_layer_call_fn_26063130

inputs
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
	unknown_2:
жМ
	unknown_3:
уМ
	unknown_4:	М
	unknown_5:	у
	unknown_6:
identityИвStatefulPartitionedCall╠
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
GPU 2J 8В *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_260629692
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
у
═
while_cond_26064885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064885___redundant_placeholder06
2while_while_cond_26064885___redundant_placeholder16
2while_while_cond_26064885___redundant_placeholder26
2while_while_cond_26064885___redundant_placeholder3
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
Ч

╠
/__inference_sequential_6_layer_call_fn_26063109

inputs
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
	unknown_2:
жМ
	unknown_3:
уМ
	unknown_4:	М
	unknown_5:	у
	unknown_6:
identityИвStatefulPartitionedCall╠
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
GPU 2J 8В *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_260625002
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
м

╙
/__inference_sequential_6_layer_call_fn_26063009
lstm_12_input
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
	unknown_2:
жМ
	unknown_3:
уМ
	unknown_4:	М
	unknown_5:	у
	unknown_6:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8В *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_260629692
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
_user_specified_namelstm_12_input
╘

э
lstm_13_while_cond_26063678,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1F
Blstm_13_while_lstm_13_while_cond_26063678___redundant_placeholder0F
Blstm_13_while_lstm_13_while_cond_26063678___redundant_placeholder1F
Blstm_13_while_lstm_13_while_cond_26063678___redundant_placeholder2F
Blstm_13_while_lstm_13_while_cond_26063678___redundant_placeholder3
lstm_13_while_identity
Ш
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
╩
·
/__inference_lstm_cell_13_layer_call_fn_26065303

inputs
states_0
states_1
unknown:
жМ
	unknown_0:
уМ
	unknown_1:	М
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
<:         у:         у:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260615702
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         у2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         у2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         у2

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
B:         ж:         у:         у: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:RN
(
_output_shapes
:         у
"
_user_specified_name
states/0:RN
(
_output_shapes
:         у
"
_user_specified_name
states/1
у
═
while_cond_26064361
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064361___redundant_placeholder06
2while_while_cond_26064361___redundant_placeholder16
2while_while_cond_26064361___redundant_placeholder26
2while_while_cond_26064361___redundant_placeholder3
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
╪
I
-__inference_dropout_12_layer_call_fn_26064451

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
:         ж* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260622952
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ж:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
у
═
while_cond_26060953
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26060953___redundant_placeholder06
2while_while_cond_26060953___redundant_placeholder16
2while_while_cond_26060953___redundant_placeholder26
2while_while_cond_26060953___redundant_placeholder3
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
№	
╩
&__inference_signature_wrapper_26063088
lstm_12_input
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
	unknown_2:
жМ
	unknown_3:
уМ
	unknown_4:	М
	unknown_5:	у
	unknown_6:
identityИвStatefulPartitionedCallм
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_260608652
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
_user_specified_namelstm_12_input
Ы
э
J__inference_sequential_6_layer_call_and_return_conditional_losses_26062500

inputs#
lstm_12_26062283:	]Ш$
lstm_12_26062285:
жШ
lstm_12_26062287:	Ш$
lstm_13_26062448:
жМ$
lstm_13_26062450:
уМ
lstm_13_26062452:	М#
dense_6_26062494:	у
dense_6_26062496:
identityИвdense_6/StatefulPartitionedCallвlstm_12/StatefulPartitionedCallвlstm_13/StatefulPartitionedCallо
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_26062283lstm_12_26062285lstm_12_26062287*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260622822!
lstm_12/StatefulPartitionedCallГ
dropout_12/PartitionedCallPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260622952
dropout_12/PartitionedCall╦
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0lstm_13_26062448lstm_13_26062450lstm_13_26062452*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260624472!
lstm_13/StatefulPartitionedCallГ
dropout_13/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260624602
dropout_13/PartitionedCall╢
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_6_26062494dense_6_26062496*
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
GPU 2J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_260624932!
dense_6/StatefulPartitionedCallЗ
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity┤
NoOpNoOp ^dense_6/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╘

э
lstm_13_while_cond_26063344,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1F
Blstm_13_while_lstm_13_while_cond_26063344___redundant_placeholder0F
Blstm_13_while_lstm_13_while_cond_26063344___redundant_placeholder1F
Blstm_13_while_lstm_13_while_cond_26063344___redundant_placeholder2F
Blstm_13_while_lstm_13_while_cond_26063344___redundant_placeholder3
lstm_13_while_identity
Ш
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_26062631
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26062631___redundant_placeholder06
2while_while_cond_26062631___redundant_placeholder16
2while_while_cond_26062631___redundant_placeholder26
2while_while_cond_26062631___redundant_placeholder3
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
У
Й
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26065286

inputs
states_0
states_11
matmul_readvariableop_resource:	]Ш4
 matmul_1_readvariableop_resource:
жШ.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
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
P:         ж:         ж:         ж:         ж*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ж2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ж2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ж2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ж2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ж2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ж2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ж2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ж2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ж2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ж2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ж2

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
A:         ]:         ж:         ж: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ж
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ж
"
_user_specified_name
states/1
у
═
while_cond_26062827
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26062827___redundant_placeholder06
2while_while_cond_26062827___redundant_placeholder16
2while_while_cond_26062827___redundant_placeholder26
2while_while_cond_26062827___redundant_placeholder3
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
╦F
О
E__inference_lstm_12_layer_call_and_return_conditional_losses_26061023

inputs(
lstm_cell_12_26060941:	]Ш)
lstm_cell_12_26060943:
жШ$
lstm_cell_12_26060945:	Ш
identityИв$lstm_cell_12/StatefulPartitionedCallвwhileD
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
B :ж2
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
B :ж2
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
:         ж2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
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
B :ж2
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
:         ж2	
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
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_26060941lstm_cell_12_26060943lstm_cell_12_26060945*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ж:         ж:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260609402&
$lstm_cell_12/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_26060941lstm_cell_12_26060943lstm_cell_12_26060945*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26060954*
condR
while_cond_26060953*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  ж*
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
:         ж*
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
!:                  ж2
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
!:                  ж2

Identity}
NoOpNoOp%^lstm_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  ]: : : 2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  ]
 
_user_specified_nameinputs
Ч
К
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26065352

inputs
states_0
states_12
matmul_readvariableop_resource:
жМ4
 matmul_1_readvariableop_resource:
уМ.
biasadd_readvariableop_resource:	М
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         М2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2	
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
P:         у:         у:         у:         у*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         у2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         у2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         у2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         у2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         у2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         у2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         у2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         у2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         у2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         у2

Identity_2Щ
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
B:         ж:         у:         у: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:RN
(
_output_shapes
:         у
"
_user_specified_name
states/0:RN
(
_output_shapes
:         у
"
_user_specified_name
states/1
К\
Я
E__inference_lstm_13_layer_call_and_return_conditional_losses_26062447

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
жМA
-lstm_cell_13_matmul_1_readvariableop_resource:
уМ;
,lstm_cell_13_biasadd_readvariableop_resource:	М
identityИв#lstm_cell_13/BiasAdd/ReadVariableOpв"lstm_cell_13/MatMul/ReadVariableOpв$lstm_cell_13/MatMul_1/ReadVariableOpвwhileD
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
B :у2
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
B :у2
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
:         у2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
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
B :у2
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
:         у2	
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
:         ж2
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
valueB"    ж  27
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
:         ж*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOpн
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul╝
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOpй
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul_1а
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/add┤
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOpн
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimў
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
lstm_cell_13/splitЗ
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/SigmoidЛ
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_1М
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
lstm_cell_13/ReluЭ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_1Т
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/add_1Л
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/Relu_1б
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26062363*
condR
while_cond_26062362*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         у*
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
:         у*
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
:         у2
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
:         у2

Identity╚
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ж: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
╘

э
lstm_12_while_cond_26063523,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1F
Blstm_12_while_lstm_12_while_cond_26063523___redundant_placeholder0F
Blstm_12_while_lstm_12_while_cond_26063523___redundant_placeholder1F
Blstm_12_while_lstm_12_while_cond_26063523___redundant_placeholder2F
Blstm_12_while_lstm_12_while_cond_26063523___redundant_placeholder3
lstm_12_while_identity
Ш
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
у
╗
*__inference_lstm_13_layer_call_fn_26064495
inputs_0
unknown:
жМ
	unknown_0:
уМ
	unknown_1:	М
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260618632
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  у2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ж: : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:                  ж
"
_user_specified_name
inputs/0
у
═
while_cond_26064210
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064210___redundant_placeholder06
2while_while_cond_26064210___redundant_placeholder16
2while_while_cond_26064210___redundant_placeholder26
2while_while_cond_26064210___redundant_placeholder3
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
╢
╕
*__inference_lstm_12_layer_call_fn_26063842

inputs
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260629122
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ж2

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
И┤
╤	
#__inference__wrapped_model_26060865
lstm_12_inputS
@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]ШV
Bsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
жШP
Asequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	ШT
@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resource:
жМV
Bsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
уМP
Asequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	МI
6sequential_6_dense_6_tensordot_readvariableop_resource:	уB
4sequential_6_dense_6_biasadd_readvariableop_resource:
identityИв+sequential_6/dense_6/BiasAdd/ReadVariableOpв-sequential_6/dense_6/Tensordot/ReadVariableOpв8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpв7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOpв9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpвsequential_6/lstm_12/whileв8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpв7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOpв9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpвsequential_6/lstm_13/whileu
sequential_6/lstm_12/ShapeShapelstm_12_input*
T0*
_output_shapes
:2
sequential_6/lstm_12/ShapeЮ
(sequential_6/lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_6/lstm_12/strided_slice/stackв
*sequential_6/lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_12/strided_slice/stack_1в
*sequential_6/lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_12/strided_slice/stack_2р
"sequential_6/lstm_12/strided_sliceStridedSlice#sequential_6/lstm_12/Shape:output:01sequential_6/lstm_12/strided_slice/stack:output:03sequential_6/lstm_12/strided_slice/stack_1:output:03sequential_6/lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_6/lstm_12/strided_sliceЗ
 sequential_6/lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2"
 sequential_6/lstm_12/zeros/mul/y└
sequential_6/lstm_12/zeros/mulMul+sequential_6/lstm_12/strided_slice:output:0)sequential_6/lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_12/zeros/mulЙ
!sequential_6/lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_6/lstm_12/zeros/Less/y╗
sequential_6/lstm_12/zeros/LessLess"sequential_6/lstm_12/zeros/mul:z:0*sequential_6/lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_12/zeros/LessН
#sequential_6/lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ж2%
#sequential_6/lstm_12/zeros/packed/1╫
!sequential_6/lstm_12/zeros/packedPack+sequential_6/lstm_12/strided_slice:output:0,sequential_6/lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_6/lstm_12/zeros/packedЙ
 sequential_6/lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_6/lstm_12/zeros/Const╩
sequential_6/lstm_12/zerosFill*sequential_6/lstm_12/zeros/packed:output:0)sequential_6/lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:         ж2
sequential_6/lstm_12/zerosЛ
"sequential_6/lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2$
"sequential_6/lstm_12/zeros_1/mul/y╞
 sequential_6/lstm_12/zeros_1/mulMul+sequential_6/lstm_12/strided_slice:output:0+sequential_6/lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_12/zeros_1/mulН
#sequential_6/lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_6/lstm_12/zeros_1/Less/y├
!sequential_6/lstm_12/zeros_1/LessLess$sequential_6/lstm_12/zeros_1/mul:z:0,sequential_6/lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_6/lstm_12/zeros_1/LessС
%sequential_6/lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ж2'
%sequential_6/lstm_12/zeros_1/packed/1▌
#sequential_6/lstm_12/zeros_1/packedPack+sequential_6/lstm_12/strided_slice:output:0.sequential_6/lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_6/lstm_12/zeros_1/packedН
"sequential_6/lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_6/lstm_12/zeros_1/Const╥
sequential_6/lstm_12/zeros_1Fill,sequential_6/lstm_12/zeros_1/packed:output:0+sequential_6/lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ж2
sequential_6/lstm_12/zeros_1Я
#sequential_6/lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_6/lstm_12/transpose/perm└
sequential_6/lstm_12/transpose	Transposelstm_12_input,sequential_6/lstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2 
sequential_6/lstm_12/transposeО
sequential_6/lstm_12/Shape_1Shape"sequential_6/lstm_12/transpose:y:0*
T0*
_output_shapes
:2
sequential_6/lstm_12/Shape_1в
*sequential_6/lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_12/strided_slice_1/stackж
,sequential_6/lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_1/stack_1ж
,sequential_6/lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_1/stack_2ь
$sequential_6/lstm_12/strided_slice_1StridedSlice%sequential_6/lstm_12/Shape_1:output:03sequential_6/lstm_12/strided_slice_1/stack:output:05sequential_6/lstm_12/strided_slice_1/stack_1:output:05sequential_6/lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_1п
0sequential_6/lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         22
0sequential_6/lstm_12/TensorArrayV2/element_shapeЖ
"sequential_6/lstm_12/TensorArrayV2TensorListReserve9sequential_6/lstm_12/TensorArrayV2/element_shape:output:0-sequential_6/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_6/lstm_12/TensorArrayV2щ
Jsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2L
Jsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape╠
<sequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_6/lstm_12/transpose:y:0Ssequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensorв
*sequential_6/lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_12/strided_slice_2/stackж
,sequential_6/lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_2/stack_1ж
,sequential_6/lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_2/stack_2·
$sequential_6/lstm_12/strided_slice_2StridedSlice"sequential_6/lstm_12/transpose:y:03sequential_6/lstm_12/strided_slice_2/stack:output:05sequential_6/lstm_12/strided_slice_2/stack_1:output:05sequential_6/lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_2Ї
7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype029
7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOpБ
(sequential_6/lstm_12/lstm_cell_12/MatMulMatMul-sequential_6/lstm_12/strided_slice_2:output:0?sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2*
(sequential_6/lstm_12/lstm_cell_12/MatMul√
9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpBsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02;
9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp¤
*sequential_6/lstm_12/lstm_cell_12/MatMul_1MatMul#sequential_6/lstm_12/zeros:output:0Asequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2,
*sequential_6/lstm_12/lstm_cell_12/MatMul_1Ї
%sequential_6/lstm_12/lstm_cell_12/addAddV22sequential_6/lstm_12/lstm_cell_12/MatMul:product:04sequential_6/lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2'
%sequential_6/lstm_12/lstm_cell_12/addє
8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpAsequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02:
8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpБ
)sequential_6/lstm_12/lstm_cell_12/BiasAddBiasAdd)sequential_6/lstm_12/lstm_cell_12/add:z:0@sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2+
)sequential_6/lstm_12/lstm_cell_12/BiasAddи
1sequential_6/lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_6/lstm_12/lstm_cell_12/split/split_dim╦
'sequential_6/lstm_12/lstm_cell_12/splitSplit:sequential_6/lstm_12/lstm_cell_12/split/split_dim:output:02sequential_6/lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2)
'sequential_6/lstm_12/lstm_cell_12/split╞
)sequential_6/lstm_12/lstm_cell_12/SigmoidSigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2+
)sequential_6/lstm_12/lstm_cell_12/Sigmoid╩
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_1Sigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2-
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_1р
%sequential_6/lstm_12/lstm_cell_12/mulMul/sequential_6/lstm_12/lstm_cell_12/Sigmoid_1:y:0%sequential_6/lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:         ж2'
%sequential_6/lstm_12/lstm_cell_12/mul╜
&sequential_6/lstm_12/lstm_cell_12/ReluRelu0sequential_6/lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2(
&sequential_6/lstm_12/lstm_cell_12/Reluё
'sequential_6/lstm_12/lstm_cell_12/mul_1Mul-sequential_6/lstm_12/lstm_cell_12/Sigmoid:y:04sequential_6/lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2)
'sequential_6/lstm_12/lstm_cell_12/mul_1ц
'sequential_6/lstm_12/lstm_cell_12/add_1AddV2)sequential_6/lstm_12/lstm_cell_12/mul:z:0+sequential_6/lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2)
'sequential_6/lstm_12/lstm_cell_12/add_1╩
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_2Sigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2-
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_2╝
(sequential_6/lstm_12/lstm_cell_12/Relu_1Relu+sequential_6/lstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2*
(sequential_6/lstm_12/lstm_cell_12/Relu_1ї
'sequential_6/lstm_12/lstm_cell_12/mul_2Mul/sequential_6/lstm_12/lstm_cell_12/Sigmoid_2:y:06sequential_6/lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2)
'sequential_6/lstm_12/lstm_cell_12/mul_2╣
2sequential_6/lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  24
2sequential_6/lstm_12/TensorArrayV2_1/element_shapeМ
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
sequential_6/lstm_12/timeй
-sequential_6/lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2/
-sequential_6/lstm_12/while/maximum_iterationsФ
'sequential_6/lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/lstm_12/while/loop_counter╬
sequential_6/lstm_12/whileWhile0sequential_6/lstm_12/while/loop_counter:output:06sequential_6/lstm_12/while/maximum_iterations:output:0"sequential_6/lstm_12/time:output:0-sequential_6/lstm_12/TensorArrayV2_1:handle:0#sequential_6/lstm_12/zeros:output:0%sequential_6/lstm_12/zeros_1:output:0-sequential_6/lstm_12/strided_slice_1:output:0Lsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resourceBsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resourceAsequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_6_lstm_12_while_body_26060605*4
cond,R*
(sequential_6_lstm_12_while_cond_26060604*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
sequential_6/lstm_12/while▀
Esequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2G
Esequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape╜
7sequential_6/lstm_12/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_6/lstm_12/while:output:3Nsequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ж*
element_dtype029
7sequential_6/lstm_12/TensorArrayV2Stack/TensorListStackл
*sequential_6/lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2,
*sequential_6/lstm_12/strided_slice_3/stackж
,sequential_6/lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_6/lstm_12/strided_slice_3/stack_1ж
,sequential_6/lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_3/stack_2Щ
$sequential_6/lstm_12/strided_slice_3StridedSlice@sequential_6/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/lstm_12/strided_slice_3/stack:output:05sequential_6/lstm_12/strided_slice_3/stack_1:output:05sequential_6/lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ж*
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_3г
%sequential_6/lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_6/lstm_12/transpose_1/perm·
 sequential_6/lstm_12/transpose_1	Transpose@sequential_6/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_6/lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ж2"
 sequential_6/lstm_12/transpose_1Р
sequential_6/lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_6/lstm_12/runtimeн
 sequential_6/dropout_12/IdentityIdentity$sequential_6/lstm_12/transpose_1:y:0*
T0*,
_output_shapes
:         ж2"
 sequential_6/dropout_12/IdentityС
sequential_6/lstm_13/ShapeShape)sequential_6/dropout_12/Identity:output:0*
T0*
_output_shapes
:2
sequential_6/lstm_13/ShapeЮ
(sequential_6/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_6/lstm_13/strided_slice/stackв
*sequential_6/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_13/strided_slice/stack_1в
*sequential_6/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_13/strided_slice/stack_2р
"sequential_6/lstm_13/strided_sliceStridedSlice#sequential_6/lstm_13/Shape:output:01sequential_6/lstm_13/strided_slice/stack:output:03sequential_6/lstm_13/strided_slice/stack_1:output:03sequential_6/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_6/lstm_13/strided_sliceЗ
 sequential_6/lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2"
 sequential_6/lstm_13/zeros/mul/y└
sequential_6/lstm_13/zeros/mulMul+sequential_6/lstm_13/strided_slice:output:0)sequential_6/lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_13/zeros/mulЙ
!sequential_6/lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2#
!sequential_6/lstm_13/zeros/Less/y╗
sequential_6/lstm_13/zeros/LessLess"sequential_6/lstm_13/zeros/mul:z:0*sequential_6/lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_13/zeros/LessН
#sequential_6/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :у2%
#sequential_6/lstm_13/zeros/packed/1╫
!sequential_6/lstm_13/zeros/packedPack+sequential_6/lstm_13/strided_slice:output:0,sequential_6/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_6/lstm_13/zeros/packedЙ
 sequential_6/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_6/lstm_13/zeros/Const╩
sequential_6/lstm_13/zerosFill*sequential_6/lstm_13/zeros/packed:output:0)sequential_6/lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:         у2
sequential_6/lstm_13/zerosЛ
"sequential_6/lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2$
"sequential_6/lstm_13/zeros_1/mul/y╞
 sequential_6/lstm_13/zeros_1/mulMul+sequential_6/lstm_13/strided_slice:output:0+sequential_6/lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_13/zeros_1/mulН
#sequential_6/lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :ш2%
#sequential_6/lstm_13/zeros_1/Less/y├
!sequential_6/lstm_13/zeros_1/LessLess$sequential_6/lstm_13/zeros_1/mul:z:0,sequential_6/lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_6/lstm_13/zeros_1/LessС
%sequential_6/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :у2'
%sequential_6/lstm_13/zeros_1/packed/1▌
#sequential_6/lstm_13/zeros_1/packedPack+sequential_6/lstm_13/strided_slice:output:0.sequential_6/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_6/lstm_13/zeros_1/packedН
"sequential_6/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_6/lstm_13/zeros_1/Const╥
sequential_6/lstm_13/zeros_1Fill,sequential_6/lstm_13/zeros_1/packed:output:0+sequential_6/lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:         у2
sequential_6/lstm_13/zeros_1Я
#sequential_6/lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_6/lstm_13/transpose/perm▌
sequential_6/lstm_13/transpose	Transpose)sequential_6/dropout_12/Identity:output:0,sequential_6/lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:         ж2 
sequential_6/lstm_13/transposeО
sequential_6/lstm_13/Shape_1Shape"sequential_6/lstm_13/transpose:y:0*
T0*
_output_shapes
:2
sequential_6/lstm_13/Shape_1в
*sequential_6/lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_13/strided_slice_1/stackж
,sequential_6/lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_1/stack_1ж
,sequential_6/lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_1/stack_2ь
$sequential_6/lstm_13/strided_slice_1StridedSlice%sequential_6/lstm_13/Shape_1:output:03sequential_6/lstm_13/strided_slice_1/stack:output:05sequential_6/lstm_13/strided_slice_1/stack_1:output:05sequential_6/lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_1п
0sequential_6/lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         22
0sequential_6/lstm_13/TensorArrayV2/element_shapeЖ
"sequential_6/lstm_13/TensorArrayV2TensorListReserve9sequential_6/lstm_13/TensorArrayV2/element_shape:output:0-sequential_6/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_6/lstm_13/TensorArrayV2щ
Jsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2L
Jsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape╠
<sequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_6/lstm_13/transpose:y:0Ssequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensorв
*sequential_6/lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_13/strided_slice_2/stackж
,sequential_6/lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_2/stack_1ж
,sequential_6/lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_2/stack_2√
$sequential_6/lstm_13/strided_slice_2StridedSlice"sequential_6/lstm_13/transpose:y:03sequential_6/lstm_13/strided_slice_2/stack:output:05sequential_6/lstm_13/strided_slice_2/stack_1:output:05sequential_6/lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ж*
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_2ї
7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype029
7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOpБ
(sequential_6/lstm_13/lstm_cell_13/MatMulMatMul-sequential_6/lstm_13/strided_slice_2:output:0?sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2*
(sequential_6/lstm_13/lstm_cell_13/MatMul√
9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpBsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02;
9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¤
*sequential_6/lstm_13/lstm_cell_13/MatMul_1MatMul#sequential_6/lstm_13/zeros:output:0Asequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2,
*sequential_6/lstm_13/lstm_cell_13/MatMul_1Ї
%sequential_6/lstm_13/lstm_cell_13/addAddV22sequential_6/lstm_13/lstm_cell_13/MatMul:product:04sequential_6/lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2'
%sequential_6/lstm_13/lstm_cell_13/addє
8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpAsequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02:
8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpБ
)sequential_6/lstm_13/lstm_cell_13/BiasAddBiasAdd)sequential_6/lstm_13/lstm_cell_13/add:z:0@sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2+
)sequential_6/lstm_13/lstm_cell_13/BiasAddи
1sequential_6/lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_6/lstm_13/lstm_cell_13/split/split_dim╦
'sequential_6/lstm_13/lstm_cell_13/splitSplit:sequential_6/lstm_13/lstm_cell_13/split/split_dim:output:02sequential_6/lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2)
'sequential_6/lstm_13/lstm_cell_13/split╞
)sequential_6/lstm_13/lstm_cell_13/SigmoidSigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2+
)sequential_6/lstm_13/lstm_cell_13/Sigmoid╩
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_1Sigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2-
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_1р
%sequential_6/lstm_13/lstm_cell_13/mulMul/sequential_6/lstm_13/lstm_cell_13/Sigmoid_1:y:0%sequential_6/lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:         у2'
%sequential_6/lstm_13/lstm_cell_13/mul╜
&sequential_6/lstm_13/lstm_cell_13/ReluRelu0sequential_6/lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2(
&sequential_6/lstm_13/lstm_cell_13/Reluё
'sequential_6/lstm_13/lstm_cell_13/mul_1Mul-sequential_6/lstm_13/lstm_cell_13/Sigmoid:y:04sequential_6/lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2)
'sequential_6/lstm_13/lstm_cell_13/mul_1ц
'sequential_6/lstm_13/lstm_cell_13/add_1AddV2)sequential_6/lstm_13/lstm_cell_13/mul:z:0+sequential_6/lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2)
'sequential_6/lstm_13/lstm_cell_13/add_1╩
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_2Sigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2-
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_2╝
(sequential_6/lstm_13/lstm_cell_13/Relu_1Relu+sequential_6/lstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2*
(sequential_6/lstm_13/lstm_cell_13/Relu_1ї
'sequential_6/lstm_13/lstm_cell_13/mul_2Mul/sequential_6/lstm_13/lstm_cell_13/Sigmoid_2:y:06sequential_6/lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2)
'sequential_6/lstm_13/lstm_cell_13/mul_2╣
2sequential_6/lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   24
2sequential_6/lstm_13/TensorArrayV2_1/element_shapeМ
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
sequential_6/lstm_13/timeй
-sequential_6/lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2/
-sequential_6/lstm_13/while/maximum_iterationsФ
'sequential_6/lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/lstm_13/while/loop_counter╬
sequential_6/lstm_13/whileWhile0sequential_6/lstm_13/while/loop_counter:output:06sequential_6/lstm_13/while/maximum_iterations:output:0"sequential_6/lstm_13/time:output:0-sequential_6/lstm_13/TensorArrayV2_1:handle:0#sequential_6/lstm_13/zeros:output:0%sequential_6/lstm_13/zeros_1:output:0-sequential_6/lstm_13/strided_slice_1:output:0Lsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resourceBsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resourceAsequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_6_lstm_13_while_body_26060753*4
cond,R*
(sequential_6_lstm_13_while_cond_26060752*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
sequential_6/lstm_13/while▀
Esequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2G
Esequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape╜
7sequential_6/lstm_13/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_6/lstm_13/while:output:3Nsequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         у*
element_dtype029
7sequential_6/lstm_13/TensorArrayV2Stack/TensorListStackл
*sequential_6/lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2,
*sequential_6/lstm_13/strided_slice_3/stackж
,sequential_6/lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_6/lstm_13/strided_slice_3/stack_1ж
,sequential_6/lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_3/stack_2Щ
$sequential_6/lstm_13/strided_slice_3StridedSlice@sequential_6/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/lstm_13/strided_slice_3/stack:output:05sequential_6/lstm_13/strided_slice_3/stack_1:output:05sequential_6/lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         у*
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_3г
%sequential_6/lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_6/lstm_13/transpose_1/perm·
 sequential_6/lstm_13/transpose_1	Transpose@sequential_6/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_6/lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:         у2"
 sequential_6/lstm_13/transpose_1Р
sequential_6/lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_6/lstm_13/runtimeн
 sequential_6/dropout_13/IdentityIdentity$sequential_6/lstm_13/transpose_1:y:0*
T0*,
_output_shapes
:         у2"
 sequential_6/dropout_13/Identity╓
-sequential_6/dense_6/Tensordot/ReadVariableOpReadVariableOp6sequential_6_dense_6_tensordot_readvariableop_resource*
_output_shapes
:	у*
dtype02/
-sequential_6/dense_6/Tensordot/ReadVariableOpФ
#sequential_6/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_6/dense_6/Tensordot/axesЫ
#sequential_6/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_6/dense_6/Tensordot/freeе
$sequential_6/dense_6/Tensordot/ShapeShape)sequential_6/dropout_13/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_6/dense_6/Tensordot/ShapeЮ
,sequential_6/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/dense_6/Tensordot/GatherV2/axis║
'sequential_6/dense_6/Tensordot/GatherV2GatherV2-sequential_6/dense_6/Tensordot/Shape:output:0,sequential_6/dense_6/Tensordot/free:output:05sequential_6/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_6/dense_6/Tensordot/GatherV2в
.sequential_6/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_6/dense_6/Tensordot/GatherV2_1/axis└
)sequential_6/dense_6/Tensordot/GatherV2_1GatherV2-sequential_6/dense_6/Tensordot/Shape:output:0,sequential_6/dense_6/Tensordot/axes:output:07sequential_6/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_6/dense_6/Tensordot/GatherV2_1Ц
$sequential_6/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_6/dense_6/Tensordot/Const╘
#sequential_6/dense_6/Tensordot/ProdProd0sequential_6/dense_6/Tensordot/GatherV2:output:0-sequential_6/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_6/dense_6/Tensordot/ProdЪ
&sequential_6/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_6/dense_6/Tensordot/Const_1▄
%sequential_6/dense_6/Tensordot/Prod_1Prod2sequential_6/dense_6/Tensordot/GatherV2_1:output:0/sequential_6/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_6/dense_6/Tensordot/Prod_1Ъ
*sequential_6/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_6/dense_6/Tensordot/concat/axisЩ
%sequential_6/dense_6/Tensordot/concatConcatV2,sequential_6/dense_6/Tensordot/free:output:0,sequential_6/dense_6/Tensordot/axes:output:03sequential_6/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/dense_6/Tensordot/concatр
$sequential_6/dense_6/Tensordot/stackPack,sequential_6/dense_6/Tensordot/Prod:output:0.sequential_6/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_6/dense_6/Tensordot/stackє
(sequential_6/dense_6/Tensordot/transpose	Transpose)sequential_6/dropout_13/Identity:output:0.sequential_6/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:         у2*
(sequential_6/dense_6/Tensordot/transposeє
&sequential_6/dense_6/Tensordot/ReshapeReshape,sequential_6/dense_6/Tensordot/transpose:y:0-sequential_6/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2(
&sequential_6/dense_6/Tensordot/ReshapeЄ
%sequential_6/dense_6/Tensordot/MatMulMatMul/sequential_6/dense_6/Tensordot/Reshape:output:05sequential_6/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2'
%sequential_6/dense_6/Tensordot/MatMulЪ
&sequential_6/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_6/dense_6/Tensordot/Const_2Ю
,sequential_6/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/dense_6/Tensordot/concat_1/axisж
'sequential_6/dense_6/Tensordot/concat_1ConcatV20sequential_6/dense_6/Tensordot/GatherV2:output:0/sequential_6/dense_6/Tensordot/Const_2:output:05sequential_6/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_6/dense_6/Tensordot/concat_1ф
sequential_6/dense_6/TensordotReshape/sequential_6/dense_6/Tensordot/MatMul:product:00sequential_6/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2 
sequential_6/dense_6/Tensordot╦
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_6/dense_6/BiasAdd/ReadVariableOp█
sequential_6/dense_6/BiasAddBiasAdd'sequential_6/dense_6/Tensordot:output:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
sequential_6/dense_6/BiasAddд
sequential_6/dense_6/SoftmaxSoftmax%sequential_6/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:         2
sequential_6/dense_6/SoftmaxЕ
IdentityIdentity&sequential_6/dense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╚
NoOpNoOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp.^sequential_6/dense_6/Tensordot/ReadVariableOp9^sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp8^sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:^sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^sequential_6/lstm_12/while9^sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp8^sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:^sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^sequential_6/lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2Z
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
:         ]
'
_user_specified_namelstm_12_input
╢
╕
*__inference_lstm_12_layer_call_fn_26063831

inputs
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260622822
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ж2

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
┤?
╓
while_body_26062363
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
жМI
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	М
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
жМG
3while_lstm_cell_13_matmul_1_readvariableop_resource:
уМA
2while_lstm_cell_13_biasadd_readvariableop_resource:	МИв)while/lstm_cell_13/BiasAdd/ReadVariableOpв(while/lstm_cell_13/MatMul/ReadVariableOpв*while/lstm_cell_13/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp╫
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul╨
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp└
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul_1╕
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/add╚
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp┼
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/BiasAddК
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dimП
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
while/lstm_cell_13/splitЩ
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/SigmoidЭ
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_1б
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mulР
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu╡
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_1к
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/add_1Э
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_2П
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu_1╣
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2V
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
р
║
*__inference_lstm_12_layer_call_fn_26063809
inputs_0
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
identityИвStatefulPartitionedCallТ
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:                  ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260610232
StatefulPartitionedCallЙ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:                  ж2

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
┤?
╓
while_body_26064735
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
жМI
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	М
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
жМG
3while_lstm_cell_13_matmul_1_readvariableop_resource:
уМA
2while_lstm_cell_13_biasadd_readvariableop_resource:	МИв)while/lstm_cell_13/BiasAdd/ReadVariableOpв(while/lstm_cell_13/MatMul/ReadVariableOpв*while/lstm_cell_13/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp╫
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul╨
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp└
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul_1╕
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/add╚
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp┼
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/BiasAddК
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dimП
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
while/lstm_cell_13/splitЩ
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/SigmoidЭ
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_1б
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mulР
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu╡
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_1к
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/add_1Э
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_2П
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu_1╣
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2V
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
├L
╓
!__inference__traced_save_26065506
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
ShardedFilenameь
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*■
valueЇBё"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names╠
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices┐
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableop@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableop@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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

identity_1Identity_1:output:0*Х
_input_shapesГ
А: :	у:: : : : : :	]Ш:
жШ:Ш:
жМ:
уМ:М: : : : :	у::	]Ш:
жШ:Ш:
жМ:
уМ:М:	у::	]Ш:
жШ:Ш:
жМ:
уМ:М: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	у: 
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
:	]Ш:&	"
 
_output_shapes
:
жШ:!


_output_shapes	
:Ш:&"
 
_output_shapes
:
жМ:&"
 
_output_shapes
:
уМ:!

_output_shapes	
:М:
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
:	у: 

_output_shapes
::%!

_output_shapes
:	]Ш:&"
 
_output_shapes
:
жШ:!

_output_shapes	
:Ш:&"
 
_output_shapes
:
жМ:&"
 
_output_shapes
:
уМ:!

_output_shapes	
:М:%!

_output_shapes
:	у: 

_output_shapes
::%!

_output_shapes
:	]Ш:&"
 
_output_shapes
:
жШ:!

_output_shapes	
:Ш:&"
 
_output_shapes
:
жМ:& "
 
_output_shapes
:
уМ:!!

_output_shapes	
:М:"

_output_shapes
: 
К\
Я
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064970

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
жМA
-lstm_cell_13_matmul_1_readvariableop_resource:
уМ;
,lstm_cell_13_biasadd_readvariableop_resource:	М
identityИв#lstm_cell_13/BiasAdd/ReadVariableOpв"lstm_cell_13/MatMul/ReadVariableOpв$lstm_cell_13/MatMul_1/ReadVariableOpвwhileD
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
B :у2
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
B :у2
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
:         у2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
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
B :у2
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
:         у2	
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
:         ж2
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
valueB"    ж  27
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
:         ж*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOpн
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul╝
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOpй
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul_1а
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/add┤
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOpн
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimў
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
lstm_cell_13/splitЗ
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/SigmoidЛ
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_1М
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
lstm_cell_13/ReluЭ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_1Т
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/add_1Л
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/Relu_1б
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064886*
condR
while_cond_26064885*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         у*
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
:         у*
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
:         у2
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
:         у2

Identity╚
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ж: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
у
═
while_cond_26065036
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26065036___redundant_placeholder06
2while_while_cond_26065036___redundant_placeholder16
2while_while_cond_26065036___redundant_placeholder26
2while_while_cond_26065036___redundant_placeholder3
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
хJ
╘

lstm_12_while_body_26063197,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]ШQ
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШK
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorL
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]ШO
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:
жШI
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpв0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpв2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp╙
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeГ
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         ]*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItemс
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpў
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2#
!lstm_12/while/lstm_cell_12/MatMulш
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpр
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2%
#lstm_12/while/lstm_cell_12/MatMul_1╪
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2 
lstm_12/while/lstm_cell_12/addр
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpх
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2$
"lstm_12/while/lstm_cell_12/BiasAddЪ
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dimп
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2"
 lstm_12/while/lstm_cell_12/split▒
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2$
"lstm_12/while/lstm_cell_12/Sigmoid╡
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2&
$lstm_12/while/lstm_cell_12/Sigmoid_1┴
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*(
_output_shapes
:         ж2 
lstm_12/while/lstm_cell_12/mulи
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2!
lstm_12/while/lstm_cell_12/Relu╒
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2"
 lstm_12/while/lstm_cell_12/mul_1╩
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2"
 lstm_12/while/lstm_cell_12/add_1╡
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2&
$lstm_12/while/lstm_cell_12/Sigmoid_2з
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2#
!lstm_12/while/lstm_cell_12/Relu_1┘
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2"
 lstm_12/while/lstm_cell_12/mul_2И
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
lstm_12/while/add/yЙ
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
lstm_12/while/add_1/yЮ
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1Л
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identityж
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1Н
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2║
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3о
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:         ж2
lstm_12/while/Identity_4о
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0^lstm_12/while/NoOp*
T0*(
_output_shapes
:         ж2
lstm_12/while/Identity_5Ж
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
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"╚
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2f
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
┤?
╓
while_body_26062632
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
жМI
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	М
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
жМG
3while_lstm_cell_13_matmul_1_readvariableop_resource:
уМA
2while_lstm_cell_13_biasadd_readvariableop_resource:	МИв)while/lstm_cell_13/BiasAdd/ReadVariableOpв(while/lstm_cell_13/MatMul/ReadVariableOpв*while/lstm_cell_13/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp╫
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul╨
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp└
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul_1╕
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/add╚
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp┼
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/BiasAddК
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dimП
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
while/lstm_cell_13/splitЩ
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/SigmoidЭ
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_1б
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mulР
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu╡
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_1к
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/add_1Э
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_2П
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu_1╣
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2V
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
╨F
П
E__inference_lstm_13_layer_call_and_return_conditional_losses_26061653

inputs)
lstm_cell_13_26061571:
жМ)
lstm_cell_13_26061573:
уМ$
lstm_cell_13_26061575:	М
identityИв$lstm_cell_13/StatefulPartitionedCallвwhileD
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
B :у2
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
B :у2
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
:         у2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
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
B :у2
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
:         у2	
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
!:                  ж2
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
valueB"    ж  27
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
:         ж*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_26061571lstm_cell_13_26061573lstm_cell_13_26061575*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         у:         у:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260615702&
$lstm_cell_13/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_26061571lstm_cell_13_26061573lstm_cell_13_26061575*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26061584*
condR
while_cond_26061583*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  у*
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
:         у*
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
!:                  у2
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
!:                  у2

Identity}
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ж: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ж
 
_user_specified_nameinputs
Д\
Ю
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064446

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]ШA
-lstm_cell_12_matmul_1_readvariableop_resource:
жШ;
,lstm_cell_12_biasadd_readvariableop_resource:	Ш
identityИв#lstm_cell_12/BiasAdd/ReadVariableOpв"lstm_cell_12/MatMul/ReadVariableOpв$lstm_cell_12/MatMul_1/ReadVariableOpвwhileD
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
B :ж2
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
B :ж2
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
:         ж2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
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
B :ж2
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
:         ж2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOpн
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul╝
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOpй
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul_1а
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/add┤
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOpн
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimў
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
lstm_cell_12/splitЗ
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/SigmoidЛ
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_1М
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
lstm_cell_12/ReluЭ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_1Т
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/add_1Л
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Relu_1б
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064362*
condR
while_cond_26064361*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ж*
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
:         ж*
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
:         ж2
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
:         ж2

Identity╚
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╞Т
К
$__inference__traced_restore_26065615
file_prefix2
assignvariableop_dense_6_kernel:	у-
assignvariableop_1_dense_6_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_12_lstm_cell_12_kernel:	]ШL
8assignvariableop_8_lstm_12_lstm_cell_12_recurrent_kernel:
жШ;
,assignvariableop_9_lstm_12_lstm_cell_12_bias:	ШC
/assignvariableop_10_lstm_13_lstm_cell_13_kernel:
жМM
9assignvariableop_11_lstm_13_lstm_cell_13_recurrent_kernel:
уМ<
-assignvariableop_12_lstm_13_lstm_cell_13_bias:	М#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: <
)assignvariableop_17_adam_dense_6_kernel_m:	у5
'assignvariableop_18_adam_dense_6_bias_m:I
6assignvariableop_19_adam_lstm_12_lstm_cell_12_kernel_m:	]ШT
@assignvariableop_20_adam_lstm_12_lstm_cell_12_recurrent_kernel_m:
жШC
4assignvariableop_21_adam_lstm_12_lstm_cell_12_bias_m:	ШJ
6assignvariableop_22_adam_lstm_13_lstm_cell_13_kernel_m:
жМT
@assignvariableop_23_adam_lstm_13_lstm_cell_13_recurrent_kernel_m:
уМC
4assignvariableop_24_adam_lstm_13_lstm_cell_13_bias_m:	М<
)assignvariableop_25_adam_dense_6_kernel_v:	у5
'assignvariableop_26_adam_dense_6_bias_v:I
6assignvariableop_27_adam_lstm_12_lstm_cell_12_kernel_v:	]ШT
@assignvariableop_28_adam_lstm_12_lstm_cell_12_recurrent_kernel_v:
жШC
4assignvariableop_29_adam_lstm_12_lstm_cell_12_bias_v:	ШJ
6assignvariableop_30_adam_lstm_13_lstm_cell_13_kernel_v:
жМT
@assignvariableop_31_adam_lstm_13_lstm_cell_13_recurrent_kernel_v:
уМC
4assignvariableop_32_adam_lstm_13_lstm_cell_13_bias_v:	М
identity_34ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Є
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*■
valueЇBё"B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
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

IdentityЮ
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_12_lstm_cell_12_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╜
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_12_lstm_cell_12_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9▒
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_12_lstm_cell_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10╖
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_13_lstm_cell_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11┴
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_13_lstm_cell_13_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12╡
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_13_lstm_cell_13_biasIdentity_12:output:0"/device:CPU:0*
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
Identity_17▒
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_6_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18п
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_6_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╛
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_12_lstm_cell_12_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20╚
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_12_lstm_cell_12_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21╝
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_12_lstm_cell_12_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22╛
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_lstm_13_lstm_cell_13_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23╚
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_lstm_13_lstm_cell_13_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24╝
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_13_lstm_cell_13_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25▒
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_6_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26п
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_6_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27╛
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_12_lstm_cell_12_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28╚
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_lstm_12_lstm_cell_12_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29╝
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_12_lstm_cell_12_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30╛
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_13_lstm_cell_13_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31╚
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_13_lstm_cell_13_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32╝
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_13_lstm_cell_13_bias_vIdentity_32:output:0"/device:CPU:0*
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
Л
З
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26061086

inputs

states
states_11
matmul_readvariableop_resource:	]Ш4
 matmul_1_readvariableop_resource:
жШ.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
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
P:         ж:         ж:         ж:         ж*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ж2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ж2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ж2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ж2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ж2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ж2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ж2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ж2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ж2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ж2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ж2

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
A:         ]:         ж:         ж: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:PL
(
_output_shapes
:         ж
 
_user_specified_namestates:PL
(
_output_shapes
:         ж
 
_user_specified_namestates
╟
∙
/__inference_lstm_cell_12_layer_call_fn_26065205

inputs
states_0
states_1
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
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
<:         ж:         ж:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260609402
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ж2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ж2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ж2

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
A:         ]:         ж:         ж: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ж
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ж
"
_user_specified_name
states/1
╟
∙
/__inference_lstm_cell_12_layer_call_fn_26065222

inputs
states_0
states_1
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
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
<:         ж:         ж:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260610862
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ж2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         ж2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         ж2

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
A:         ]:         ж:         ж: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ж
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ж
"
_user_specified_name
states/1
щJ
╓

lstm_13_while_body_26063345,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0O
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
жМQ
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМK
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	М
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorM
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
жМO
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
уМI
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	МИв1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpв0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpв2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp╙
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeД
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItemт
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpў
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2#
!lstm_13/while/lstm_cell_13/MatMulш
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpр
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2%
#lstm_13/while/lstm_cell_13/MatMul_1╪
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2 
lstm_13/while/lstm_cell_13/addр
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpх
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2$
"lstm_13/while/lstm_cell_13/BiasAddЪ
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dimп
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2"
 lstm_13/while/lstm_cell_13/split▒
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2$
"lstm_13/while/lstm_cell_13/Sigmoid╡
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2&
$lstm_13/while/lstm_cell_13/Sigmoid_1┴
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:         у2 
lstm_13/while/lstm_cell_13/mulи
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2!
lstm_13/while/lstm_cell_13/Relu╒
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2"
 lstm_13/while/lstm_cell_13/mul_1╩
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2"
 lstm_13/while/lstm_cell_13/add_1╡
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2&
$lstm_13/while/lstm_cell_13/Sigmoid_2з
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2#
!lstm_13/while/lstm_cell_13/Relu_1┘
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2"
 lstm_13/while/lstm_cell_13/mul_2И
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
lstm_13/while/add/yЙ
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
lstm_13/while/add_1/yЮ
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1Л
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identityж
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1Н
lstm_13/while/Identity_2Identitylstm_13/while/add:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2║
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3о
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:         у2
lstm_13/while/Identity_4о
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:         у2
lstm_13/while/Identity_5Ж
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
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"╚
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2f
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
╫
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_26065148

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         у2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         у*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         у2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         у2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         у2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         у2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         у:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
┤?
╓
while_body_26064886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
жМI
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	М
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
жМG
3while_lstm_cell_13_matmul_1_readvariableop_resource:
уМA
2while_lstm_cell_13_biasadd_readvariableop_resource:	МИв)while/lstm_cell_13/BiasAdd/ReadVariableOpв(while/lstm_cell_13/MatMul/ReadVariableOpв*while/lstm_cell_13/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp╫
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul╨
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp└
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul_1╕
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/add╚
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp┼
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/BiasAddК
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dimП
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
while/lstm_cell_13/splitЩ
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/SigmoidЭ
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_1б
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mulР
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu╡
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_1к
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/add_1Э
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_2П
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu_1╣
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2V
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
┤?
╓
while_body_26064584
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0G
3while_lstm_cell_13_matmul_readvariableop_resource_0:
жМI
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	М
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorE
1while_lstm_cell_13_matmul_readvariableop_resource:
жМG
3while_lstm_cell_13_matmul_1_readvariableop_resource:
уМA
2while_lstm_cell_13_biasadd_readvariableop_resource:	МИв)while/lstm_cell_13/BiasAdd/ReadVariableOpв(while/lstm_cell_13/MatMul/ReadVariableOpв*while/lstm_cell_13/MatMul_1/ReadVariableOp├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem╩
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp╫
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul╨
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOp└
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/MatMul_1╕
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/add╚
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOp┼
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
while/lstm_cell_13/BiasAddК
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dimП
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
while/lstm_cell_13/splitЩ
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/SigmoidЭ
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_1б
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mulР
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu╡
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_1к
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/add_1Э
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Sigmoid_2П
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/Relu_1╣
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
while/lstm_cell_13/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2V
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
╗
f
-__inference_dropout_12_layer_call_fn_26064456

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
:         ж* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260627452
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         ж2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ж22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
у
═
while_cond_26063908
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26063908___redundant_placeholder06
2while_while_cond_26063908___redundant_placeholder16
2while_while_cond_26063908___redundant_placeholder26
2while_while_cond_26063908___redundant_placeholder3
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
у
═
while_cond_26061163
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26061163___redundant_placeholder06
2while_while_cond_26061163___redundant_placeholder16
2while_while_cond_26061163___redundant_placeholder26
2while_while_cond_26061163___redundant_placeholder3
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
░?
╘
while_body_26062828
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]ШI
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]ШG
3while_lstm_cell_12_matmul_1_readvariableop_resource:
жШA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв)while/lstm_cell_12/BiasAdd/ReadVariableOpв(while/lstm_cell_12/MatMul/ReadVariableOpв*while/lstm_cell_12/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp╫
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul╨
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp└
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul_1╕
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/add╚
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp┼
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/BiasAddК
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dimП
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
while/lstm_cell_12/splitЩ
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/SigmoidЭ
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_1б
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mulР
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu╡
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_1к
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/add_1Э
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_2П
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu_1╣
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2V
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
ФМ
З
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063798

inputsF
3lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]ШI
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:
жШC
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	ШG
3lstm_13_lstm_cell_13_matmul_readvariableop_resource:
жМI
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
уМC
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	М<
)dense_6_tensordot_readvariableop_resource:	у5
'dense_6_biasadd_readvariableop_resource:
identityИвdense_6/BiasAdd/ReadVariableOpв dense_6/Tensordot/ReadVariableOpв+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpв*lstm_12/lstm_cell_12/MatMul/ReadVariableOpв,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpвlstm_12/whileв+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpв*lstm_13/lstm_cell_13/MatMul/ReadVariableOpв,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpвlstm_13/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/ShapeД
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stackИ
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1И
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2Т
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
B :ж2
lstm_12/zeros/mul/yМ
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
B :ш2
lstm_12/zeros/Less/yЗ
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
B :ж2
lstm_12/zeros/packed/1г
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
lstm_12/zeros/ConstЦ
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*(
_output_shapes
:         ж2
lstm_12/zerosq
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
lstm_12/zeros_1/mul/yТ
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
B :ш2
lstm_12/zeros_1/Less/yП
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
B :ж2
lstm_12/zeros_1/packed/1й
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
lstm_12/zeros_1/ConstЮ
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*(
_output_shapes
:         ж2
lstm_12/zeros_1Е
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/permТ
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:         ]2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1И
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stackМ
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1М
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2Ю
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1Х
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_12/TensorArrayV2/element_shape╥
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2╧
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ]   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensorИ
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stackМ
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1М
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2м
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         ]*
shrink_axis_mask2
lstm_12/strided_slice_2═
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOp═
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_12/lstm_cell_12/MatMul╘
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp╔
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_12/lstm_cell_12/MatMul_1└
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
lstm_12/lstm_cell_12/add╠
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp═
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_12/lstm_cell_12/BiasAddО
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dimЧ
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
lstm_12/lstm_cell_12/splitЯ
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/Sigmoidг
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2 
lstm_12/lstm_cell_12/Sigmoid_1м
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/mulЦ
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/Relu╜
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/mul_1▓
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/add_1г
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2 
lstm_12/lstm_cell_12/Sigmoid_2Х
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/Relu_1┴
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
lstm_12/lstm_cell_12/mul_2Я
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2'
%lstm_12/TensorArrayV2_1/element_shape╪
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
lstm_12/timeП
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counterЛ
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_12_while_body_26063524*'
condR
lstm_12_while_cond_26063523*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
lstm_12/while┼
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ж*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStackС
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_12/strided_slice_3/stackМ
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1М
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2╦
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ж*
shrink_axis_mask2
lstm_12/strided_slice_3Й
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/perm╞
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*,
_output_shapes
:         ж2
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
 *лкк?2
dropout_12/dropout/Constк
dropout_12/dropout/MulMullstm_12/transpose_1:y:0!dropout_12/dropout/Const:output:0*
T0*,
_output_shapes
:         ж2
dropout_12/dropout/Mul{
dropout_12/dropout/ShapeShapelstm_12/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shape┌
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*,
_output_shapes
:         ж*
dtype021
/dropout_12/dropout/random_uniform/RandomUniformЛ
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А>2#
!dropout_12/dropout/GreaterEqual/yя
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         ж2!
dropout_12/dropout/GreaterEqualе
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ж2
dropout_12/dropout/Castл
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*,
_output_shapes
:         ж2
dropout_12/dropout/Mul_1j
lstm_13/ShapeShapedropout_12/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_13/ShapeД
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stackИ
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1И
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2Т
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
B :у2
lstm_13/zeros/mul/yМ
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
B :ш2
lstm_13/zeros/Less/yЗ
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
B :у2
lstm_13/zeros/packed/1г
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
lstm_13/zeros/ConstЦ
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:         у2
lstm_13/zerosq
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
lstm_13/zeros_1/mul/yТ
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
B :ш2
lstm_13/zeros_1/Less/yП
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
B :у2
lstm_13/zeros_1/packed/1й
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
lstm_13/zeros_1/ConstЮ
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:         у2
lstm_13/zeros_1Е
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/permй
lstm_13/transpose	Transposedropout_12/dropout/Mul_1:z:0lstm_13/transpose/perm:output:0*
T0*,
_output_shapes
:         ж2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1И
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stackМ
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1М
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2Ю
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1Х
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         2%
#lstm_13/TensorArrayV2/element_shape╥
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2╧
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeШ
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensorИ
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stackМ
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1М
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2н
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         ж*
shrink_axis_mask2
lstm_13/strided_slice_2╬
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOp═
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_13/lstm_cell_13/MatMul╘
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp╔
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_13/lstm_cell_13/MatMul_1└
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
lstm_13/lstm_cell_13/add╠
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp═
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_13/lstm_cell_13/BiasAddО
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dimЧ
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
lstm_13/lstm_cell_13/splitЯ
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/Sigmoidг
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2 
lstm_13/lstm_cell_13/Sigmoid_1м
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/mulЦ
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/Relu╜
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/mul_1▓
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/add_1г
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2 
lstm_13/lstm_cell_13/Sigmoid_2Х
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/Relu_1┴
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
lstm_13/lstm_cell_13/mul_2Я
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2'
%lstm_13/TensorArrayV2_1/element_shape╪
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
lstm_13/timeП
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counterЛ
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_13_while_body_26063679*'
condR
lstm_13_while_cond_26063678*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
lstm_13/while┼
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeЙ
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         у*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStackС
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         2
lstm_13/strided_slice_3/stackМ
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1М
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2╦
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:         у*
shrink_axis_mask2
lstm_13/strided_slice_3Й
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/perm╞
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:         у2
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
 *  а?2
dropout_13/dropout/Constк
dropout_13/dropout/MulMullstm_13/transpose_1:y:0!dropout_13/dropout/Const:output:0*
T0*,
_output_shapes
:         у2
dropout_13/dropout/Mul{
dropout_13/dropout/ShapeShapelstm_13/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_13/dropout/Shape┌
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*,
_output_shapes
:         у*
dtype021
/dropout_13/dropout/random_uniform/RandomUniformЛ
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_13/dropout/GreaterEqual/yя
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         у2!
dropout_13/dropout/GreaterEqualе
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         у2
dropout_13/dropout/Castл
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*,
_output_shapes
:         у2
dropout_13/dropout/Mul_1п
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	у*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axesБ
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
dense_6/Tensordot/ShapeД
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axis∙
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2И
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axis 
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
dense_6/Tensordot/Constа
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/ProdА
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1и
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1А
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axis╪
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concatм
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack┐
dense_6/Tensordot/transpose	Transposedropout_13/dropout/Mul_1:z:0!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:         у2
dense_6/Tensordot/transpose┐
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  2
dense_6/Tensordot/Reshape╛
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_6/Tensordot/MatMulА
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/Const_2Д
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axisх
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1░
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         2
dense_6/Tensordotд
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOpз
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         2
dense_6/BiasAdd}
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*+
_output_shapes
:         2
dense_6/Softmaxx
IdentityIdentitydense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:         2

Identity╞
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2@
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
:         ]
 
_user_specified_nameinputs
у
═
while_cond_26062197
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26062197___redundant_placeholder06
2while_while_cond_26062197___redundant_placeholder16
2while_while_cond_26062197___redundant_placeholder26
2while_while_cond_26062197___redundant_placeholder3
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
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
█
ё
(sequential_6_lstm_13_while_cond_26060752F
Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counterL
Hsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations*
&sequential_6_lstm_13_while_placeholder,
(sequential_6_lstm_13_while_placeholder_1,
(sequential_6_lstm_13_while_placeholder_2,
(sequential_6_lstm_13_while_placeholder_3H
Dsequential_6_lstm_13_while_less_sequential_6_lstm_13_strided_slice_1`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26060752___redundant_placeholder0`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26060752___redundant_placeholder1`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26060752___redundant_placeholder2`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26060752___redundant_placeholder3'
#sequential_6_lstm_13_while_identity
┘
sequential_6/lstm_13/while/LessLess&sequential_6_lstm_13_while_placeholderDsequential_6_lstm_13_while_less_sequential_6_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_6/lstm_13/while/LessЬ
#sequential_6/lstm_13/while/IdentityIdentity#sequential_6/lstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_6/lstm_13/while/Identity"S
#sequential_6_lstm_13_while_identity,sequential_6/lstm_13/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
╫
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_26062549

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constx
dropout/MulMulinputsdropout/Const:output:0*
T0*,
_output_shapes
:         у2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         у*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y├
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:         у2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         у2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         у2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         у2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         у:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
у
═
while_cond_26061583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26061583___redundant_placeholder06
2while_while_cond_26061583___redundant_placeholder16
2while_while_cond_26061583___redundant_placeholder26
2while_while_cond_26061583___redundant_placeholder3
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
К\
Я
E__inference_lstm_13_layer_call_and_return_conditional_losses_26065121

inputs?
+lstm_cell_13_matmul_readvariableop_resource:
жМA
-lstm_cell_13_matmul_1_readvariableop_resource:
уМ;
,lstm_cell_13_biasadd_readvariableop_resource:	М
identityИв#lstm_cell_13/BiasAdd/ReadVariableOpв"lstm_cell_13/MatMul/ReadVariableOpв$lstm_cell_13/MatMul_1/ReadVariableOpвwhileD
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
B :у2
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
B :у2
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
:         у2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
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
B :у2
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
:         у2	
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
:         ж2
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
valueB"    ж  27
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
:         ж*
shrink_axis_mask2
strided_slice_2╢
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource* 
_output_shapes
:
жМ*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOpн
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul╝
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
уМ*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOpй
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/MatMul_1а
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/add┤
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:М*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOpн
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dimў
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2
lstm_cell_13/splitЗ
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/SigmoidЛ
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_1М
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2
lstm_cell_13/ReluЭ
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_1Т
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/add_1Л
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/Relu_1б
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2
lstm_cell_13/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26065037*
condR
while_cond_26065036*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         у*
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
:         у*
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
:         у2
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
:         у2

Identity╚
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ж: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
░]
ў
(sequential_6_lstm_13_while_body_26060753F
Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counterL
Hsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations*
&sequential_6_lstm_13_while_placeholder,
(sequential_6_lstm_13_while_placeholder_1,
(sequential_6_lstm_13_while_placeholder_2,
(sequential_6_lstm_13_while_placeholder_3E
Asequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1_0Б
}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0\
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:
жМ^
Jsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
уМX
Isequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	М'
#sequential_6_lstm_13_while_identity)
%sequential_6_lstm_13_while_identity_1)
%sequential_6_lstm_13_while_identity_2)
%sequential_6_lstm_13_while_identity_3)
%sequential_6_lstm_13_while_identity_4)
%sequential_6_lstm_13_while_identity_5C
?sequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1
{sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensorZ
Fsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:
жМ\
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
уМV
Gsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	МИв>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpв=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpв?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpэ
Lsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2N
Lsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape╥
>sequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0&sequential_6_lstm_13_while_placeholderUsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02@
>sequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItemЙ
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpHsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0* 
_output_shapes
:
жМ*
dtype02?
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpл
.sequential_6/lstm_13/while/lstm_cell_13/MatMulMatMulEsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М20
.sequential_6/lstm_13/while/lstm_cell_13/MatMulП
?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpJsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
уМ*
dtype02A
?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpФ
0sequential_6/lstm_13/while/lstm_cell_13/MatMul_1MatMul(sequential_6_lstm_13_while_placeholder_2Gsequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М22
0sequential_6/lstm_13/while/lstm_cell_13/MatMul_1М
+sequential_6/lstm_13/while/lstm_cell_13/addAddV28sequential_6/lstm_13/while/lstm_cell_13/MatMul:product:0:sequential_6/lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:         М2-
+sequential_6/lstm_13/while/lstm_cell_13/addЗ
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpIsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:М*
dtype02@
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpЩ
/sequential_6/lstm_13/while/lstm_cell_13/BiasAddBiasAdd/sequential_6/lstm_13/while/lstm_cell_13/add:z:0Fsequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         М21
/sequential_6/lstm_13/while/lstm_cell_13/BiasAdd┤
7sequential_6/lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_6/lstm_13/while/lstm_cell_13/split/split_dimу
-sequential_6/lstm_13/while/lstm_cell_13/splitSplit@sequential_6/lstm_13/while/lstm_cell_13/split/split_dim:output:08sequential_6/lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:         у:         у:         у:         у*
	num_split2/
-sequential_6/lstm_13/while/lstm_cell_13/split╪
/sequential_6/lstm_13/while/lstm_cell_13/SigmoidSigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:         у21
/sequential_6/lstm_13/while/lstm_cell_13/Sigmoid▄
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:         у23
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1ї
+sequential_6/lstm_13/while/lstm_cell_13/mulMul5sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1:y:0(sequential_6_lstm_13_while_placeholder_3*
T0*(
_output_shapes
:         у2-
+sequential_6/lstm_13/while/lstm_cell_13/mul╧
,sequential_6/lstm_13/while/lstm_cell_13/ReluRelu6sequential_6/lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:         у2.
,sequential_6/lstm_13/while/lstm_cell_13/ReluЙ
-sequential_6/lstm_13/while/lstm_cell_13/mul_1Mul3sequential_6/lstm_13/while/lstm_cell_13/Sigmoid:y:0:sequential_6/lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:         у2/
-sequential_6/lstm_13/while/lstm_cell_13/mul_1■
-sequential_6/lstm_13/while/lstm_cell_13/add_1AddV2/sequential_6/lstm_13/while/lstm_cell_13/mul:z:01sequential_6/lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:         у2/
-sequential_6/lstm_13/while/lstm_cell_13/add_1▄
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:         у23
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2╬
.sequential_6/lstm_13/while/lstm_cell_13/Relu_1Relu1sequential_6/lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:         у20
.sequential_6/lstm_13/while/lstm_cell_13/Relu_1Н
-sequential_6/lstm_13/while/lstm_cell_13/mul_2Mul5sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2:y:0<sequential_6/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:         у2/
-sequential_6/lstm_13/while/lstm_cell_13/mul_2╔
?sequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_6_lstm_13_while_placeholder_1&sequential_6_lstm_13_while_placeholder1sequential_6/lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItemЖ
 sequential_6/lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_6/lstm_13/while/add/y╜
sequential_6/lstm_13/while/addAddV2&sequential_6_lstm_13_while_placeholder)sequential_6/lstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_13/while/addК
"sequential_6/lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_6/lstm_13/while/add_1/y▀
 sequential_6/lstm_13/while/add_1AddV2Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counter+sequential_6/lstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_13/while/add_1┐
#sequential_6/lstm_13/while/IdentityIdentity$sequential_6/lstm_13/while/add_1:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_6/lstm_13/while/Identityч
%sequential_6/lstm_13/while/Identity_1IdentityHsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_1┴
%sequential_6/lstm_13/while/Identity_2Identity"sequential_6/lstm_13/while/add:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_2ю
%sequential_6/lstm_13/while/Identity_3IdentityOsequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_3т
%sequential_6/lstm_13/while/Identity_4Identity1sequential_6/lstm_13/while/lstm_cell_13/mul_2:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*(
_output_shapes
:         у2'
%sequential_6/lstm_13/while/Identity_4т
%sequential_6/lstm_13/while/Identity_5Identity1sequential_6/lstm_13/while/lstm_cell_13/add_1:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*(
_output_shapes
:         у2'
%sequential_6/lstm_13/while/Identity_5╟
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
%sequential_6_lstm_13_while_identity_5.sequential_6/lstm_13/while/Identity_5:output:0"Ф
Gsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resourceIsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"Ц
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resourceJsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"Т
Fsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resourceHsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"Д
?sequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1Asequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1_0"№
{sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2А
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2~
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2В
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
И&
ї
while_body_26061794
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_01
while_lstm_cell_13_26061818_0:
жМ1
while_lstm_cell_13_26061820_0:
уМ,
while_lstm_cell_13_26061822_0:	М
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor/
while_lstm_cell_13_26061818:
жМ/
while_lstm_cell_13_26061820:
уМ*
while_lstm_cell_13_26061822:	МИв*while/lstm_cell_13/StatefulPartitionedCall├
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  29
7while/TensorArrayV2Read/TensorListGetItem/element_shape╘
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:         ж*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemь
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_26061818_0while_lstm_cell_13_26061820_0while_lstm_cell_13_26061822_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         у:         у:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260617162,
*while/lstm_cell_13/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         у2
while/Identity_5З

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
while_lstm_cell_13_26061818while_lstm_cell_13_26061818_0"<
while_lstm_cell_13_26061820while_lstm_cell_13_26061820_0"<
while_lstm_cell_13_26061822while_lstm_cell_13_26061822_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         у:         у: : : : : 2X
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
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
: 
╩
·
/__inference_lstm_cell_13_layer_call_fn_26065320

inputs
states_0
states_1
unknown:
жМ
	unknown_0:
уМ
	unknown_1:	М
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
<:         у:         у:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260617162
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         у2

IdentityА

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:         у2

Identity_1А

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:         у2

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
B:         ж:         у:         у: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ж
 
_user_specified_nameinputs:RN
(
_output_shapes
:         у
"
_user_specified_name
states/0:RN
(
_output_shapes
:         у
"
_user_specified_name
states/1
Й
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_26062460

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:         у2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:         у2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         у:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
╣
╣
*__inference_lstm_13_layer_call_fn_26064517

inputs
unknown:
жМ
	unknown_0:
уМ
	unknown_1:	М
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260627162
StatefulPartitionedCallА
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:         у2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:         ж: : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
Е&
є
while_body_26060954
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_12_26060978_0:	]Ш1
while_lstm_cell_12_26060980_0:
жШ,
while_lstm_cell_12_26060982_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_12_26060978:	]Ш/
while_lstm_cell_12_26060980:
жШ*
while_lstm_cell_12_26060982:	ШИв*while/lstm_cell_12/StatefulPartitionedCall├
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
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_26060978_0while_lstm_cell_12_26060980_0while_lstm_cell_12_26060982_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         ж:         ж:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260609402,
*while/lstm_cell_12/StatefulPartitionedCallў
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3е
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_4е
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_5З

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
while_lstm_cell_12_26060978while_lstm_cell_12_26060978_0"<
while_lstm_cell_12_26060980while_lstm_cell_12_26060980_0"<
while_lstm_cell_12_26060982while_lstm_cell_12_26060982_0"0
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2X
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
█
ё
(sequential_6_lstm_12_while_cond_26060604F
Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counterL
Hsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations*
&sequential_6_lstm_12_while_placeholder,
(sequential_6_lstm_12_while_placeholder_1,
(sequential_6_lstm_12_while_placeholder_2,
(sequential_6_lstm_12_while_placeholder_3H
Dsequential_6_lstm_12_while_less_sequential_6_lstm_12_strided_slice_1`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26060604___redundant_placeholder0`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26060604___redundant_placeholder1`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26060604___redundant_placeholder2`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26060604___redundant_placeholder3'
#sequential_6_lstm_12_while_identity
┘
sequential_6/lstm_12/while/LessLess&sequential_6_lstm_12_while_placeholderDsequential_6_lstm_12_while_less_sequential_6_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_6/lstm_12/while/LessЬ
#sequential_6/lstm_12/while/IdentityIdentity#sequential_6/lstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_6/lstm_12/while/Identity"S
#sequential_6_lstm_12_while_identity,sequential_6/lstm_12/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :         ж:         ж: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
:
м

╙
/__inference_sequential_6_layer_call_fn_26062519
lstm_12_input
unknown:	]Ш
	unknown_0:
жШ
	unknown_1:	Ш
	unknown_2:
жМ
	unknown_3:
уМ
	unknown_4:	М
	unknown_5:	у
	unknown_6:
identityИвStatefulPartitionedCall╙
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
GPU 2J 8В *S
fNRL
J__inference_sequential_6_layer_call_and_return_conditional_losses_260625002
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
_user_specified_namelstm_12_input
Д\
Ю
E__inference_lstm_12_layer_call_and_return_conditional_losses_26062912

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]ШA
-lstm_cell_12_matmul_1_readvariableop_resource:
жШ;
,lstm_cell_12_biasadd_readvariableop_resource:	Ш
identityИв#lstm_cell_12/BiasAdd/ReadVariableOpв"lstm_cell_12/MatMul/ReadVariableOpв$lstm_cell_12/MatMul_1/ReadVariableOpвwhileD
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
B :ж2
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
B :ж2
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
:         ж2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
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
B :ж2
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
:         ж2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOpн
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul╝
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOpй
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul_1а
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/add┤
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOpн
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimў
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
lstm_cell_12/splitЗ
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/SigmoidЛ
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_1М
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
lstm_cell_12/ReluЭ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_1Т
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/add_1Л
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Relu_1б
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26062828*
condR
while_cond_26062827*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ж*
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
:         ж*
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
:         ж2
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
:         ж2

Identity╚
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
░?
╘
while_body_26062198
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]ШI
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]ШG
3while_lstm_cell_12_matmul_1_readvariableop_resource:
жШA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв)while/lstm_cell_12/BiasAdd/ReadVariableOpв(while/lstm_cell_12/MatMul/ReadVariableOpв*while/lstm_cell_12/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp╫
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul╨
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp└
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul_1╕
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/add╚
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp┼
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/BiasAddК
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dimП
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
while/lstm_cell_12/splitЩ
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/SigmoidЭ
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_1б
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mulР
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu╡
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_1к
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/add_1Э
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_2П
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu_1╣
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2V
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
░?
╘
while_body_26064060
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]ШI
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]ШG
3while_lstm_cell_12_matmul_1_readvariableop_resource:
жШA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв)while/lstm_cell_12/BiasAdd/ReadVariableOpв(while/lstm_cell_12/MatMul/ReadVariableOpв*while/lstm_cell_12/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp╫
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul╨
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp└
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul_1╕
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/add╚
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp┼
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/BiasAddК
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dimП
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
while/lstm_cell_12/splitЩ
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/SigmoidЭ
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_1б
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mulР
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu╡
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_1к
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/add_1Э
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_2П
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu_1╣
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2V
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
░
Ї
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063034
lstm_12_input#
lstm_12_26063012:	]Ш$
lstm_12_26063014:
жШ
lstm_12_26063016:	Ш$
lstm_13_26063020:
жМ$
lstm_13_26063022:
уМ
lstm_13_26063024:	М#
dense_6_26063028:	у
dense_6_26063030:
identityИвdense_6/StatefulPartitionedCallвlstm_12/StatefulPartitionedCallвlstm_13/StatefulPartitionedCall╡
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_26063012lstm_12_26063014lstm_12_26063016*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260622822!
lstm_12/StatefulPartitionedCallГ
dropout_12/PartitionedCallPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ж* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260622952
dropout_12/PartitionedCall╦
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0lstm_13_26063020lstm_13_26063022lstm_13_26063024*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260624472!
lstm_13/StatefulPartitionedCallГ
dropout_13/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         у* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260624602
dropout_13/PartitionedCall╢
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_6_26063028dense_6_26063030*
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
GPU 2J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_260624932!
dense_6/StatefulPartitionedCallЗ
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         2

Identity┤
NoOpNoOp ^dense_6/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         ]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:         ]
'
_user_specified_namelstm_12_input
Ж
Ш
*__inference_dense_6_layer_call_fn_26065157

inputs
unknown:	у
	unknown_0:
identityИвStatefulPartitionedCall∙
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
GPU 2J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_260624932
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
_construction_contextkEagerRuntime*/
_input_shapes
:         у: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
░?
╘
while_body_26064362
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]ШI
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:
жШC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	Ш
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]ШG
3while_lstm_cell_12_matmul_1_readvariableop_resource:
жШA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ШИв)while/lstm_cell_12/BiasAdd/ReadVariableOpв(while/lstm_cell_12/MatMul/ReadVariableOpв*while/lstm_cell_12/MatMul_1/ReadVariableOp├
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]Ш*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp╫
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul╨
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0* 
_output_shapes
:
жШ*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOp└
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/MatMul_1╕
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/add╚
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:Ш*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOp┼
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
while/lstm_cell_12/BiasAddК
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dimП
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
while/lstm_cell_12/splitЩ
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/SigmoidЭ
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_1б
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mulР
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu╡
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_1к
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/add_1Э
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Sigmoid_2П
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/Relu_1╣
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
while/lstm_cell_12/mul_2р
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
while/Identity_2Ъ
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3О
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_4О
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:         ж2
while/Identity_5▐

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
while_strided_slice_1while_strided_slice_1_0"и
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :         ж:         ж: : : : : 2V
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
:         ж:.*
(
_output_shapes
:         ж:

_output_shapes
: :

_output_shapes
: 
╨F
П
E__inference_lstm_13_layer_call_and_return_conditional_losses_26061863

inputs)
lstm_cell_13_26061781:
жМ)
lstm_cell_13_26061783:
уМ$
lstm_cell_13_26061785:	М
identityИв$lstm_cell_13/StatefulPartitionedCallвwhileD
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
B :у2
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
B :у2
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
:         у2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :у2
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
B :у2
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
:         у2	
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
!:                  ж2
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
valueB"    ж  27
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
:         ж*
shrink_axis_mask2
strided_slice_2и
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_26061781lstm_cell_13_26061783lstm_cell_13_26061785*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:         у:         у:         у*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260617162&
$lstm_cell_13/StatefulPartitionedCallП
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_26061781lstm_cell_13_26061783lstm_cell_13_26061785*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         у:         у: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26061794*
condR
while_cond_26061793*M
output_shapes<
:: : : : :         у:         у: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    у   22
0TensorArrayV2Stack/TensorListStack/element_shapeЄ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:                  у*
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
:         у*
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
!:                  у2
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
!:                  у2

Identity}
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':                  ж: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:                  ж
 
_user_specified_nameinputs
╫
g
H__inference_dropout_12_layer_call_and_return_conditional_losses_26062745

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
:         ж2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape╣
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:         ж*
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
:         ж2
dropout/GreaterEqualД
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:         ж2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:         ж2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:         ж2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ж:T P
,
_output_shapes
:         ж
 
_user_specified_nameinputs
у
═
while_cond_26064583
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064583___redundant_placeholder06
2while_while_cond_26064583___redundant_placeholder16
2while_while_cond_26064583___redundant_placeholder26
2while_while_cond_26064583___redundant_placeholder3
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
B: : : : :         у:         у: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:         у:.*
(
_output_shapes
:         у:

_output_shapes
: :

_output_shapes
:
Д\
Ю
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064295

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]ШA
-lstm_cell_12_matmul_1_readvariableop_resource:
жШ;
,lstm_cell_12_biasadd_readvariableop_resource:	Ш
identityИв#lstm_cell_12/BiasAdd/ReadVariableOpв"lstm_cell_12/MatMul/ReadVariableOpв$lstm_cell_12/MatMul_1/ReadVariableOpвwhileD
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
B :ж2
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
B :ж2
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
:         ж2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ж2
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
B :ж2
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
:         ж2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOpн
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul╝
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOpй
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/MatMul_1а
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/add┤
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOpн
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimў
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*d
_output_shapesR
P:         ж:         ж:         ж:         ж*
	num_split2
lstm_cell_12/splitЗ
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/SigmoidЛ
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_1М
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul~
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*(
_output_shapes
:         ж2
lstm_cell_12/ReluЭ
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_1Т
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/add_1Л
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Sigmoid_2}
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/Relu_1б
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
lstm_cell_12/mul_2П
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :         ж:         ж: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064211*
condR
while_cond_26064210*M
output_shapes<
:: : : : :         ж:         ж: : : : : *
parallel_iterations 2
while╡
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"    ж  22
0TensorArrayV2Stack/TensorListStack/element_shapeщ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:         ж*
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
:         ж*
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
:         ж2
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
:         ж2

Identity╚
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         ]
 
_user_specified_nameinputs
╘!
¤
E__inference_dense_6_layer_call_and_return_conditional_losses_26062493

inputs4
!tensordot_readvariableop_resource:	у-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpЧ
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	у*
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
Tensordot/stackС
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:         у2
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
_construction_contextkEagerRuntime*/
_input_shapes
:         у: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:         у
 
_user_specified_nameinputs
У
Й
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26065254

inputs
states_0
states_11
matmul_readvariableop_resource:	]Ш4
 matmul_1_readvariableop_resource:
жШ.
biasadd_readvariableop_resource:	Ш
identity

identity_1

identity_2ИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpвMatMul_1/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]Ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2
MatMulХ
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
жШ*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         Ш2
addН
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ш*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Ш2	
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
P:         ж:         ж:         ж:         ж*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:         ж2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:         ж2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:         ж2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:         ж2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:         ж2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:         ж2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:         ж2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:         ж2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:         ж2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ж2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:         ж2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:         ж2

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
A:         ]:         ж:         ж: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:         ]
 
_user_specified_nameinputs:RN
(
_output_shapes
:         ж
"
_user_specified_name
states/0:RN
(
_output_shapes
:         ж
"
_user_specified_name
states/1"иL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╛
serving_defaultк
K
lstm_12_input:
serving_default_lstm_12_input:0         ]?
dense_64
StatefulPartitionedCall:0         tensorflow/serving/predict:Л╝
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
trainable_variables
regularization_losses
		variables

	keras_api

signatures
А__call__
+Б&call_and_return_all_conditional_losses
В_default_save_signature"
_tf_keras_sequential
┼
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
Г__call__
+Д&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
з
trainable_variables
regularization_losses
	variables
	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
┼
cell

state_spec
trainable_variables
regularization_losses
	variables
	keras_api
З__call__
+И&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
з
trainable_variables
regularization_losses
	variables
	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"
_tf_keras_layer
╜

 kernel
!bias
"trainable_variables
#regularization_losses
$	variables
%	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
у
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
╬
1non_trainable_variables
2metrics

3layers
4layer_metrics
trainable_variables
regularization_losses
		variables
5layer_regularization_losses
А__call__
В_default_save_signature
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
7trainable_variables
8regularization_losses
9	variables
:	keras_api
О__call__
+П&call_and_return_all_conditional_losses"
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
╝
;non_trainable_variables
<metrics

=states

>layers
?layer_metrics
trainable_variables
regularization_losses
	variables
@layer_regularization_losses
Г__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Anon_trainable_variables
Bmetrics

Clayers
Dlayer_metrics
trainable_variables
regularization_losses
	variables
Elayer_regularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
у
F
state_size

.kernel
/recurrent_kernel
0bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
Р__call__
+С&call_and_return_all_conditional_losses"
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
╝
Knon_trainable_variables
Lmetrics

Mstates

Nlayers
Olayer_metrics
trainable_variables
regularization_losses
	variables
Player_regularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Qnon_trainable_variables
Rmetrics

Slayers
Tlayer_metrics
trainable_variables
regularization_losses
	variables
Ulayer_regularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
!:	у2dense_6/kernel
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
░
Vnon_trainable_variables
Wmetrics

Xlayers
Ylayer_metrics
"trainable_variables
#regularization_losses
$	variables
Zlayer_regularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.:,	]Ш2lstm_12/lstm_cell_12/kernel
9:7
жШ2%lstm_12/lstm_cell_12/recurrent_kernel
(:&Ш2lstm_12/lstm_cell_12/bias
/:-
жМ2lstm_13/lstm_cell_13/kernel
9:7
уМ2%lstm_13/lstm_cell_13/recurrent_kernel
(:&М2lstm_13/lstm_cell_13/bias
 "
trackable_list_wrapper
.
[0
\1"
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
░
]non_trainable_variables
^metrics

_layers
`layer_metrics
7trainable_variables
8regularization_losses
9	variables
alayer_regularization_losses
О__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
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
░
bnon_trainable_variables
cmetrics

dlayers
elayer_metrics
Gtrainable_variables
Hregularization_losses
I	variables
flayer_regularization_losses
Р__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
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
&:$	у2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
3:1	]Ш2"Adam/lstm_12/lstm_cell_12/kernel/m
>:<
жШ2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
-:+Ш2 Adam/lstm_12/lstm_cell_12/bias/m
4:2
жМ2"Adam/lstm_13/lstm_cell_13/kernel/m
>:<
уМ2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
-:+М2 Adam/lstm_13/lstm_cell_13/bias/m
&:$	у2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
3:1	]Ш2"Adam/lstm_12/lstm_cell_12/kernel/v
>:<
жШ2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
-:+Ш2 Adam/lstm_12/lstm_cell_12/bias/v
4:2
жМ2"Adam/lstm_13/lstm_cell_13/kernel/v
>:<
уМ2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
-:+М2 Adam/lstm_13/lstm_cell_13/bias/v
К2З
/__inference_sequential_6_layer_call_fn_26062519
/__inference_sequential_6_layer_call_fn_26063109
/__inference_sequential_6_layer_call_fn_26063130
/__inference_sequential_6_layer_call_fn_26063009└
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
Ў2є
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063457
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063798
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063034
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063059└
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
╘B╤
#__inference__wrapped_model_26060865lstm_12_input"Ш
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
Л2И
*__inference_lstm_12_layer_call_fn_26063809
*__inference_lstm_12_layer_call_fn_26063820
*__inference_lstm_12_layer_call_fn_26063831
*__inference_lstm_12_layer_call_fn_26063842╒
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
ў2Ї
E__inference_lstm_12_layer_call_and_return_conditional_losses_26063993
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064144
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064295
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064446╒
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
Ш2Х
-__inference_dropout_12_layer_call_fn_26064451
-__inference_dropout_12_layer_call_fn_26064456┤
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
╬2╦
H__inference_dropout_12_layer_call_and_return_conditional_losses_26064461
H__inference_dropout_12_layer_call_and_return_conditional_losses_26064473┤
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
Л2И
*__inference_lstm_13_layer_call_fn_26064484
*__inference_lstm_13_layer_call_fn_26064495
*__inference_lstm_13_layer_call_fn_26064506
*__inference_lstm_13_layer_call_fn_26064517╒
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
ў2Ї
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064668
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064819
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064970
E__inference_lstm_13_layer_call_and_return_conditional_losses_26065121╒
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
Ш2Х
-__inference_dropout_13_layer_call_fn_26065126
-__inference_dropout_13_layer_call_fn_26065131┤
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
╬2╦
H__inference_dropout_13_layer_call_and_return_conditional_losses_26065136
H__inference_dropout_13_layer_call_and_return_conditional_losses_26065148┤
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
╘2╤
*__inference_dense_6_layer_call_fn_26065157в
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
я2ь
E__inference_dense_6_layer_call_and_return_conditional_losses_26065188в
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
&__inference_signature_wrapper_26063088lstm_12_input"Ф
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
ж2г
/__inference_lstm_cell_12_layer_call_fn_26065205
/__inference_lstm_cell_12_layer_call_fn_26065222╛
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26065254
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26065286╛
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
/__inference_lstm_cell_13_layer_call_fn_26065303
/__inference_lstm_cell_13_layer_call_fn_26065320╛
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26065352
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26065384╛
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
 д
#__inference__wrapped_model_26060865}+,-./0 !:в7
0в-
+К(
lstm_12_input         ]
к "5к2
0
dense_6%К"
dense_6         о
E__inference_dense_6_layer_call_and_return_conditional_losses_26065188e !4в1
*в'
%К"
inputs         у
к ")в&
К
0         
Ъ Ж
*__inference_dense_6_layer_call_fn_26065157X !4в1
*в'
%К"
inputs         у
к "К         ▓
H__inference_dropout_12_layer_call_and_return_conditional_losses_26064461f8в5
.в+
%К"
inputs         ж
p 
к "*в'
 К
0         ж
Ъ ▓
H__inference_dropout_12_layer_call_and_return_conditional_losses_26064473f8в5
.в+
%К"
inputs         ж
p
к "*в'
 К
0         ж
Ъ К
-__inference_dropout_12_layer_call_fn_26064451Y8в5
.в+
%К"
inputs         ж
p 
к "К         жК
-__inference_dropout_12_layer_call_fn_26064456Y8в5
.в+
%К"
inputs         ж
p
к "К         ж▓
H__inference_dropout_13_layer_call_and_return_conditional_losses_26065136f8в5
.в+
%К"
inputs         у
p 
к "*в'
 К
0         у
Ъ ▓
H__inference_dropout_13_layer_call_and_return_conditional_losses_26065148f8в5
.в+
%К"
inputs         у
p
к "*в'
 К
0         у
Ъ К
-__inference_dropout_13_layer_call_fn_26065126Y8в5
.в+
%К"
inputs         у
p 
к "К         уК
-__inference_dropout_13_layer_call_fn_26065131Y8в5
.в+
%К"
inputs         у
p
к "К         у╒
E__inference_lstm_12_layer_call_and_return_conditional_losses_26063993Л+,-OвL
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
0                  ж
Ъ ╒
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064144Л+,-OвL
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
0                  ж
Ъ ╗
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064295r+,-?в<
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
0         ж
Ъ ╗
E__inference_lstm_12_layer_call_and_return_conditional_losses_26064446r+,-?в<
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
0         ж
Ъ м
*__inference_lstm_12_layer_call_fn_26063809~+,-OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p 

 
к "&К#                  жм
*__inference_lstm_12_layer_call_fn_26063820~+,-OвL
EвB
4Ъ1
/К,
inputs/0                  ]

 
p

 
к "&К#                  жУ
*__inference_lstm_12_layer_call_fn_26063831e+,-?в<
5в2
$К!
inputs         ]

 
p 

 
к "К         жУ
*__inference_lstm_12_layer_call_fn_26063842e+,-?в<
5в2
$К!
inputs         ]

 
p

 
к "К         ж╓
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064668М./0PвM
FвC
5Ъ2
0К-
inputs/0                  ж

 
p 

 
к "3в0
)К&
0                  у
Ъ ╓
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064819М./0PвM
FвC
5Ъ2
0К-
inputs/0                  ж

 
p

 
к "3в0
)К&
0                  у
Ъ ╝
E__inference_lstm_13_layer_call_and_return_conditional_losses_26064970s./0@в=
6в3
%К"
inputs         ж

 
p 

 
к "*в'
 К
0         у
Ъ ╝
E__inference_lstm_13_layer_call_and_return_conditional_losses_26065121s./0@в=
6в3
%К"
inputs         ж

 
p

 
к "*в'
 К
0         у
Ъ н
*__inference_lstm_13_layer_call_fn_26064484./0PвM
FвC
5Ъ2
0К-
inputs/0                  ж

 
p 

 
к "&К#                  ун
*__inference_lstm_13_layer_call_fn_26064495./0PвM
FвC
5Ъ2
0К-
inputs/0                  ж

 
p

 
к "&К#                  уФ
*__inference_lstm_13_layer_call_fn_26064506f./0@в=
6в3
%К"
inputs         ж

 
p 

 
к "К         уФ
*__inference_lstm_13_layer_call_fn_26064517f./0@в=
6в3
%К"
inputs         ж

 
p

 
к "К         у╤
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26065254В+,-Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         ж
#К 
states/1         ж
p 
к "vвs
lвi
К
0/0         ж
GЪD
 К
0/1/0         ж
 К
0/1/1         ж
Ъ ╤
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26065286В+,-Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         ж
#К 
states/1         ж
p
к "vвs
lвi
К
0/0         ж
GЪD
 К
0/1/0         ж
 К
0/1/1         ж
Ъ ж
/__inference_lstm_cell_12_layer_call_fn_26065205Є+,-Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         ж
#К 
states/1         ж
p 
к "fвc
К
0         ж
CЪ@
К
1/0         ж
К
1/1         жж
/__inference_lstm_cell_12_layer_call_fn_26065222Є+,-Вв
xвu
 К
inputs         ]
MвJ
#К 
states/0         ж
#К 
states/1         ж
p
к "fвc
К
0         ж
CЪ@
К
1/0         ж
К
1/1         ж╙
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26065352Д./0ДвА
yвv
!К
inputs         ж
MвJ
#К 
states/0         у
#К 
states/1         у
p 
к "vвs
lвi
К
0/0         у
GЪD
 К
0/1/0         у
 К
0/1/1         у
Ъ ╙
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26065384Д./0ДвА
yвv
!К
inputs         ж
MвJ
#К 
states/0         у
#К 
states/1         у
p
к "vвs
lвi
К
0/0         у
GЪD
 К
0/1/0         у
 К
0/1/1         у
Ъ и
/__inference_lstm_cell_13_layer_call_fn_26065303Ї./0ДвА
yвv
!К
inputs         ж
MвJ
#К 
states/0         у
#К 
states/1         у
p 
к "fвc
К
0         у
CЪ@
К
1/0         у
К
1/1         уи
/__inference_lstm_cell_13_layer_call_fn_26065320Ї./0ДвА
yвv
!К
inputs         ж
MвJ
#К 
states/0         у
#К 
states/1         у
p
к "fвc
К
0         у
CЪ@
К
1/0         у
К
1/1         у╟
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063034y+,-./0 !Bв?
8в5
+К(
lstm_12_input         ]
p 

 
к ")в&
К
0         
Ъ ╟
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063059y+,-./0 !Bв?
8в5
+К(
lstm_12_input         ]
p

 
к ")в&
К
0         
Ъ └
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063457r+,-./0 !;в8
1в.
$К!
inputs         ]
p 

 
к ")в&
К
0         
Ъ └
J__inference_sequential_6_layer_call_and_return_conditional_losses_26063798r+,-./0 !;в8
1в.
$К!
inputs         ]
p

 
к ")в&
К
0         
Ъ Я
/__inference_sequential_6_layer_call_fn_26062519l+,-./0 !Bв?
8в5
+К(
lstm_12_input         ]
p 

 
к "К         Я
/__inference_sequential_6_layer_call_fn_26063009l+,-./0 !Bв?
8в5
+К(
lstm_12_input         ]
p

 
к "К         Ш
/__inference_sequential_6_layer_call_fn_26063109e+,-./0 !;в8
1в.
$К!
inputs         ]
p 

 
к "К         Ш
/__inference_sequential_6_layer_call_fn_26063130e+,-./0 !;в8
1в.
$К!
inputs         ]
p

 
к "К         ╣
&__inference_signature_wrapper_26063088О+,-./0 !KвH
в 
Aк>
<
lstm_12_input+К(
lstm_12_input         ]"5к2
0
dense_6%К"
dense_6         