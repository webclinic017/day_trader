æ'
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8&
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ç*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	ç*
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

lstm_12/lstm_cell_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]ø*,
shared_namelstm_12/lstm_cell_12/kernel

/lstm_12/lstm_cell_12/kernel/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/kernel*
_output_shapes
:	]ø*
dtype0
§
%lstm_12/lstm_cell_12/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	>ø*6
shared_name'%lstm_12/lstm_cell_12/recurrent_kernel
 
9lstm_12/lstm_cell_12/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_12/lstm_cell_12/recurrent_kernel*
_output_shapes
:	>ø*
dtype0

lstm_12/lstm_cell_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ø**
shared_namelstm_12/lstm_cell_12/bias

-lstm_12/lstm_cell_12/bias/Read/ReadVariableOpReadVariableOplstm_12/lstm_cell_12/bias*
_output_shapes	
:ø*
dtype0

lstm_13/lstm_cell_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	>*,
shared_namelstm_13/lstm_cell_13/kernel

/lstm_13/lstm_cell_13/kernel/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/kernel*
_output_shapes
:	>*
dtype0
¨
%lstm_13/lstm_cell_13/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ç*6
shared_name'%lstm_13/lstm_cell_13/recurrent_kernel
¡
9lstm_13/lstm_cell_13/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_13/lstm_cell_13/recurrent_kernel* 
_output_shapes
:
ç*
dtype0

lstm_13/lstm_cell_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_13/lstm_cell_13/bias

-lstm_13/lstm_cell_13/bias/Read/ReadVariableOpReadVariableOplstm_13/lstm_cell_13/bias*
_output_shapes	
:*
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

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ç*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	ç*
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
¡
"Adam/lstm_12/lstm_cell_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]ø*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/m

6Adam/lstm_12/lstm_cell_12/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/m*
_output_shapes
:	]ø*
dtype0
µ
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	>ø*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
®
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m*
_output_shapes
:	>ø*
dtype0

 Adam/lstm_12/lstm_cell_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ø*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/m

4Adam/lstm_12/lstm_cell_12/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/m*
_output_shapes	
:ø*
dtype0
¡
"Adam/lstm_13/lstm_cell_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	>*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/m

6Adam/lstm_13/lstm_cell_13/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/m*
_output_shapes
:	>*
dtype0
¶
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ç*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
¯
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m* 
_output_shapes
:
ç*
dtype0

 Adam/lstm_13/lstm_cell_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/m

4Adam/lstm_13/lstm_cell_13/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ç*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	ç*
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
¡
"Adam/lstm_12/lstm_cell_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]ø*3
shared_name$"Adam/lstm_12/lstm_cell_12/kernel/v

6Adam/lstm_12/lstm_cell_12/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_12/lstm_cell_12/kernel/v*
_output_shapes
:	]ø*
dtype0
µ
,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	>ø*=
shared_name.,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
®
@Adam/lstm_12/lstm_cell_12/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v*
_output_shapes
:	>ø*
dtype0

 Adam/lstm_12/lstm_cell_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ø*1
shared_name" Adam/lstm_12/lstm_cell_12/bias/v

4Adam/lstm_12/lstm_cell_12/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_12/lstm_cell_12/bias/v*
_output_shapes	
:ø*
dtype0
¡
"Adam/lstm_13/lstm_cell_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	>*3
shared_name$"Adam/lstm_13/lstm_cell_13/kernel/v

6Adam/lstm_13/lstm_cell_13/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_13/lstm_cell_13/kernel/v*
_output_shapes
:	>*
dtype0
¶
,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ç*=
shared_name.,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
¯
@Adam/lstm_13/lstm_cell_13/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v* 
_output_shapes
:
ç*
dtype0

 Adam/lstm_13/lstm_cell_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_13/lstm_cell_13/bias/v

4Adam/lstm_13/lstm_cell_13/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_13/lstm_cell_13/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
ø7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*³7
value©7B¦7 B7
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
regularization_losses
trainable_variables
	variables
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
regularization_losses
trainable_variables
	variables
	keras_api
h

 kernel
!bias
"regularization_losses
#trainable_variables
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
1non_trainable_variables
2layer_regularization_losses
3metrics

4layers
trainable_variables
5layer_metrics
	variables
	regularization_losses
 

6
state_size

+kernel
,recurrent_kernel
-bias
7regularization_losses
8trainable_variables
9	variables
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
;non_trainable_variables
<layer_regularization_losses
=metrics

>layers
trainable_variables
?layer_metrics
	variables

@states
regularization_losses
 
 
 
­
Anon_trainable_variables
Blayer_regularization_losses
Cmetrics
regularization_losses
trainable_variables
Dlayer_metrics
	variables

Elayers

F
state_size

.kernel
/recurrent_kernel
0bias
Gregularization_losses
Htrainable_variables
I	variables
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
Knon_trainable_variables
Llayer_regularization_losses
Mmetrics

Nlayers
trainable_variables
Olayer_metrics
	variables

Pstates
regularization_losses
 
 
 
­
Qnon_trainable_variables
Rlayer_regularization_losses
Smetrics
regularization_losses
trainable_variables
Tlayer_metrics
	variables

Ulayers
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
­
Vnon_trainable_variables
Wlayer_regularization_losses
Xmetrics
"regularization_losses
#trainable_variables
Ylayer_metrics
$	variables

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

+0
,1
-2
­
]non_trainable_variables
^layer_regularization_losses
_metrics
7regularization_losses
8trainable_variables
`layer_metrics
9	variables

alayers
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
 

.0
/1
02

.0
/1
02
­
bnon_trainable_variables
clayer_regularization_losses
dmetrics
Gregularization_losses
Htrainable_variables
elayer_metrics
I	variables

flayers
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

VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_12/lstm_cell_12/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_12/lstm_cell_12/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_12/lstm_cell_12/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/lstm_13/lstm_cell_13/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE,Adam/lstm_13/lstm_cell_13/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_13/lstm_cell_13/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_12_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]
ª
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_12_inputlstm_12/lstm_cell_12/kernel%lstm_12/lstm_cell_12/recurrent_kernellstm_12/lstm_cell_12/biaslstm_13/lstm_cell_13/kernel%lstm_13/lstm_cell_13/recurrent_kernellstm_13/lstm_cell_13/biasdense_6/kerneldense_6/bias*
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
&__inference_signature_wrapper_26065496
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
»
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
GPU 2J 8 **
f%R#
!__inference__traced_save_26067914
¢	
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
GPU 2J 8 *-
f(R&
$__inference__traced_restore_26068023í$
Ï
g
H__inference_dropout_12_layer_call_and_return_conditional_losses_26066871

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
:ÿÿÿÿÿÿÿÿÿ>2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
°?
Ô
while_body_26065040
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	>I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	>G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
çA
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÐ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAdd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_1¡
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Reluµ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_1ª
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Relu_1¹
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
ß
Í
while_cond_26063361
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26063361___redundant_placeholder06
2while_while_cond_26063361___redundant_placeholder16
2while_while_cond_26063361___redundant_placeholder26
2while_while_cond_26063361___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
Ý
¹
*__inference_lstm_12_layer_call_fn_26066821
inputs_0
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260634312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2

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

ë
J__inference_sequential_6_layer_call_and_return_conditional_losses_26064908

inputs#
lstm_12_26064691:	]ø#
lstm_12_26064693:	>ø
lstm_12_26064695:	ø#
lstm_13_26064856:	>$
lstm_13_26064858:
ç
lstm_13_26064860:	#
dense_6_26064902:	ç
dense_6_26064904:
identity¢dense_6/StatefulPartitionedCall¢lstm_12/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall­
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_26064691lstm_12_26064693lstm_12_26064695*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260646902!
lstm_12/StatefulPartitionedCall
dropout_12/PartitionedCallPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260647032
dropout_12/PartitionedCallË
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0lstm_13_26064856lstm_13_26064858lstm_13_26064860*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260648552!
lstm_13/StatefulPartitionedCall
dropout_13/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260648682
dropout_13/PartitionedCall¶
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_6_26064902dense_6_26064904*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260649012!
dense_6/StatefulPartitionedCall
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity´
NoOpNoOp ^dense_6/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ª

Ñ
/__inference_sequential_6_layer_call_fn_26064927
lstm_12_input
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
	unknown_2:	>
	unknown_3:
ç
	unknown_4:	
	unknown_5:	ç
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_260649082
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
_user_specified_namelstm_12_input
ã
Í
while_cond_26067098
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26067098___redundant_placeholder06
2while_while_cond_26067098___redundant_placeholder16
2while_while_cond_26067098___redundant_placeholder26
2while_while_cond_26067098___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
¶
¸
*__inference_lstm_13_layer_call_fn_26067518

inputs
unknown:	>
	unknown_0:
ç
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260648552
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
°?
Ô
while_body_26067401
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	>I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	>G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
çA
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÐ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAdd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_1¡
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Reluµ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_1ª
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Relu_1¹
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
&
ó
while_body_26063992
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_13_26064016_0:	>1
while_lstm_cell_13_26064018_0:
ç,
while_lstm_cell_13_26064020_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_13_26064016:	>/
while_lstm_cell_13_26064018:
ç*
while_lstm_cell_13_26064020:	¢*while/lstm_cell_13/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_26064016_0while_lstm_cell_13_26064018_0while_lstm_cell_13_26064020_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260639782,
*while/lstm_cell_13/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_5

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
while_lstm_cell_13_26064016while_lstm_cell_13_26064016_0"<
while_lstm_cell_13_26064018while_lstm_cell_13_26064018_0"<
while_lstm_cell_13_26064020while_lstm_cell_13_26064020_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
ç[

E__inference_lstm_12_layer_call_and_return_conditional_losses_26066810

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]ø@
-lstm_cell_12_matmul_1_readvariableop_resource:	>ø;
,lstm_cell_12_biasadd_readvariableop_resource:	ø
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢whileD
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
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp­
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul»
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp©
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul_1 
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/add´
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp­
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimó
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
lstm_cell_12/split
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_1
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_1
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/add_1
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu_1 
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26066726*
condR
while_cond_26066725*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>2
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
:ÿÿÿÿÿÿÿÿÿ>2

IdentityÈ
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
º
ø
/__inference_lstm_cell_12_layer_call_fn_26067694

inputs
states_0
states_1
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
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
9:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260634942
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
states/1
Õø

J__inference_sequential_6_layer_call_and_return_conditional_losses_26065823

inputsF
3lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]øH
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:	>øC
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	øF
3lstm_13_lstm_cell_13_matmul_readvariableop_resource:	>I
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
çC
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	<
)dense_6_tensordot_readvariableop_resource:	ç5
'dense_6_biasadd_readvariableop_resource:
identity¢dense_6/BiasAdd/ReadVariableOp¢ dense_6/Tensordot/ReadVariableOp¢+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp¢*lstm_12/lstm_cell_12/MatMul/ReadVariableOp¢,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp¢lstm_12/while¢+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢*lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢lstm_13/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicel
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
lstm_12/zeros/mul/y
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
B :è2
lstm_12/zeros/Less/y
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lessr
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :>2
lstm_12/zeros/packed/1£
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
lstm_12/zeros/Const
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/zerosp
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
lstm_12/zeros_1/mul/y
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
B :è2
lstm_12/zeros_1/Less/y
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessv
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :>2
lstm_12/zeros_1/packed/1©
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
lstm_12/zeros_1/Const
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/zeros_1
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/perm
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stack
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_12/TensorArrayV2/element_shapeÒ
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2Ï
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensor
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stack
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2¬
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_12/strided_slice_2Í
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpÍ
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_12/lstm_cell_12/MatMulÓ
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpÉ
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_12/lstm_cell_12/MatMul_1À
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_12/lstm_cell_12/addÌ
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpÍ
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_12/lstm_cell_12/BiasAdd
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dim
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
lstm_12/lstm_cell_12/split
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/Sigmoid¢
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2 
lstm_12/lstm_cell_12/Sigmoid_1«
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/mul
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/Relu¼
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/mul_1±
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/add_1¢
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2 
lstm_12/lstm_cell_12/Sigmoid_2
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/Relu_1À
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/mul_2
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2'
%lstm_12/TensorArrayV2_1/element_shapeØ
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
lstm_12/time
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counter
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_12_while_body_26065563*'
condR
lstm_12_while_cond_26065562*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
lstm_12/whileÅ
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStack
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_12/strided_slice_3/stack
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2Ê
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
lstm_12/strided_slice_3
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/permÅ
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/transpose_1v
lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_12/runtime
dropout_12/IdentityIdentitylstm_12/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
dropout_12/Identityj
lstm_13/ShapeShapedropout_12/Identity:output:0*
T0*
_output_shapes
:2
lstm_13/Shape
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2
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
B :ç2
lstm_13/zeros/mul/y
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
B :è2
lstm_13/zeros/Less/y
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
B :ç2
lstm_13/zeros/packed/1£
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
lstm_13/zeros/Const
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/zerosq
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
lstm_13/zeros_1/mul/y
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
B :è2
lstm_13/zeros_1/Less/y
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
B :ç2
lstm_13/zeros_1/packed/1©
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
lstm_13/zeros_1/Const
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/zeros_1
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/perm¨
lstm_13/transpose	Transposedropout_12/Identity:output:0lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stack
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_13/TensorArrayV2/element_shapeÒ
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2Ï
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensor
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stack
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2¬
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
lstm_13/strided_slice_2Í
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpÍ
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/MatMulÔ
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpÉ
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/MatMul_1À
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/addÌ
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpÍ
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/BiasAdd
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dim
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
lstm_13/lstm_cell_13/split
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/Sigmoid£
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2 
lstm_13/lstm_cell_13/Sigmoid_1¬
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/mul
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/Relu½
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/mul_1²
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/add_1£
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2 
lstm_13/lstm_cell_13/Sigmoid_2
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/Relu_1Á
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/mul_2
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2'
%lstm_13/TensorArrayV2_1/element_shapeØ
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
lstm_13/time
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counter
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_13_while_body_26065711*'
condR
lstm_13_while_cond_26065710*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
lstm_13/whileÅ
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStack
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_13/strided_slice_3/stack
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2Ë
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
shrink_axis_mask2
lstm_13/strided_slice_3
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/permÆ
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/transpose_1v
lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_13/runtime
dropout_13/IdentityIdentitylstm_13/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dropout_13/Identity¯
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	ç*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes
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
dense_6/Tensordot/Shape
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axisù
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axisÿ
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
dense_6/Tensordot/Const 
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1¨
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axisØ
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat¬
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack¿
dense_6/Tensordot/transpose	Transposedropout_13/Identity:output:0!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dense_6/Tensordot/transpose¿
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_6/Tensordot/Reshape¾
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Tensordot/MatMul
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/Const_2
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axiså
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1°
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Tensordot¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp§
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAdd}
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Softmaxx
IdentityIdentitydense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÆ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
?
Ò
while_body_26065236
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]øH
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]øF
3while_lstm_cell_12_matmul_1_readvariableop_resource:	>øA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp×
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMulÏ
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOpÀ
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMul_1¸
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/addÈ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOpÅ
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/BiasAdd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
while/lstm_cell_12/split
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_1 
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu´
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_1©
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/add_1
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_2
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu_1¸
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
Ô
I
-__inference_dropout_12_layer_call_fn_26066876

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
:ÿÿÿÿÿÿÿÿÿ>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260647032
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
°?
Ô
while_body_26067250
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	>I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	>G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
çA
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÐ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAdd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_1¡
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Reluµ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_1ª
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Relu_1¹
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
ª

Ñ
/__inference_sequential_6_layer_call_fn_26065417
lstm_12_input
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
	unknown_2:	>
	unknown_3:
ç
	unknown_4:	
	unknown_5:	ç
	unknown_6:
identity¢StatefulPartitionedCallÓ
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_260653772
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
_user_specified_namelstm_12_input
ã
Í
while_cond_26064770
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064770___redundant_placeholder06
2while_while_cond_26064770___redundant_placeholder16
2while_while_cond_26064770___redundant_placeholder26
2while_while_cond_26064770___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
¤
µ
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065377

inputs#
lstm_12_26065355:	]ø#
lstm_12_26065357:	>ø
lstm_12_26065359:	ø#
lstm_13_26065363:	>$
lstm_13_26065365:
ç
lstm_13_26065367:	#
dense_6_26065371:	ç
dense_6_26065373:
identity¢dense_6/StatefulPartitionedCall¢"dropout_12/StatefulPartitionedCall¢"dropout_13/StatefulPartitionedCall¢lstm_12/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall­
lstm_12/StatefulPartitionedCallStatefulPartitionedCallinputslstm_12_26065355lstm_12_26065357lstm_12_26065359*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260653202!
lstm_12/StatefulPartitionedCall
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260651532$
"dropout_12/StatefulPartitionedCallÓ
lstm_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0lstm_13_26065363lstm_13_26065365lstm_13_26065367*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260651242!
lstm_13/StatefulPartitionedCallÀ
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260649572$
"dropout_13/StatefulPartitionedCall¾
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_6_26065371dense_6_26065373*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260649012!
dense_6/StatefulPartitionedCall
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityþ
NoOpNoOp ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ß
Í
while_cond_26066574
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26066574___redundant_placeholder06
2while_while_cond_26066574___redundant_placeholder16
2while_while_cond_26066574___redundant_placeholder26
2while_while_cond_26066574___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_26063571
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26063571___redundant_placeholder06
2while_while_cond_26063571___redundant_placeholder16
2while_while_cond_26063571___redundant_placeholder26
2while_while_cond_26063571___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
×
ñ
(sequential_6_lstm_12_while_cond_26063012F
Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counterL
Hsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations*
&sequential_6_lstm_12_while_placeholder,
(sequential_6_lstm_12_while_placeholder_1,
(sequential_6_lstm_12_while_placeholder_2,
(sequential_6_lstm_12_while_placeholder_3H
Dsequential_6_lstm_12_while_less_sequential_6_lstm_12_strided_slice_1`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26063012___redundant_placeholder0`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26063012___redundant_placeholder1`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26063012___redundant_placeholder2`
\sequential_6_lstm_12_while_sequential_6_lstm_12_while_cond_26063012___redundant_placeholder3'
#sequential_6_lstm_12_while_identity
Ù
sequential_6/lstm_12/while/LessLess&sequential_6_lstm_12_while_placeholderDsequential_6_lstm_12_while_less_sequential_6_lstm_12_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_6/lstm_12/while/Less
#sequential_6/lstm_12/while/IdentityIdentity#sequential_6/lstm_12/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_6/lstm_12/while/Identity"S
#sequential_6_lstm_12_while_identity,sequential_6/lstm_12/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
\

E__inference_lstm_13_layer_call_and_return_conditional_losses_26064855

inputs>
+lstm_cell_13_matmul_readvariableop_resource:	>A
-lstm_cell_13_matmul_1_readvariableop_resource:
ç;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileD
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
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2	
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
:ÿÿÿÿÿÿÿÿÿ>2
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
valueB"ÿÿÿÿ>   27
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
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul¼
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim÷
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu_1¡
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064771*
condR
while_cond_26064770*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç2
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
:ÿÿÿÿÿÿÿÿÿç2

IdentityÈ
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
Ã\
 
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067183
inputs_0>
+lstm_cell_13_matmul_readvariableop_resource:	>A
-lstm_cell_13_matmul_1_readvariableop_resource:
ç;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileF
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
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2	
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2
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
valueB"ÿÿÿÿ>   27
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
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul¼
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim÷
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu_1¡
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26067099*
condR
while_cond_26067098*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2

IdentityÈ
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
inputs/0
ß
Í
while_cond_26064605
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064605___redundant_placeholder06
2while_while_cond_26064605___redundant_placeholder16
2while_while_cond_26064605___redundant_placeholder26
2while_while_cond_26064605___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
ú%
ñ
while_body_26063362
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_12_26063386_0:	]ø0
while_lstm_cell_12_26063388_0:	>ø,
while_lstm_cell_12_26063390_0:	ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_12_26063386:	]ø.
while_lstm_cell_12_26063388:	>ø*
while_lstm_cell_12_26063390:	ø¢*while/lstm_cell_12/StatefulPartitionedCallÃ
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
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_26063386_0while_lstm_cell_12_26063388_0while_lstm_cell_12_26063390_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260633482,
*while/lstm_cell_12/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¤
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_5

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
while_lstm_cell_12_26063386while_lstm_cell_12_26063386_0"<
while_lstm_cell_12_26063388while_lstm_cell_12_26063388_0"<
while_lstm_cell_12_26063390while_lstm_cell_12_26063390_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2X
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_13_layer_call_and_return_conditional_losses_26064271

inputs(
lstm_cell_13_26064189:	>)
lstm_cell_13_26064191:
ç$
lstm_cell_13_26064193:	
identity¢$lstm_cell_13/StatefulPartitionedCall¢whileD
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
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2	
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2
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
valueB"ÿÿÿÿ>   27
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
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_26064189lstm_cell_13_26064191lstm_cell_13_26064193*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260641242&
$lstm_cell_13/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_26064189lstm_cell_13_26064191lstm_cell_13_26064193*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064202*
condR
while_cond_26064201*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2

Identity}
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
Ô!
ý
E__inference_dense_6_layer_call_and_return_conditional_losses_26067587

inputs4
!tensordot_readvariableop_resource:	ç-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ç*
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
:ÿÿÿÿÿÿÿÿÿç2
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
:ÿÿÿÿÿÿÿÿÿç: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs
Ï
g
H__inference_dropout_12_layer_call_and_return_conditional_losses_26065153

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
:ÿÿÿÿÿÿÿÿÿ>2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
ã
Í
while_cond_26066947
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26066947___redundant_placeholder06
2while_while_cond_26066947___redundant_placeholder16
2while_while_cond_26066947___redundant_placeholder26
2while_while_cond_26066947___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
¶
¸
*__inference_lstm_13_layer_call_fn_26067529

inputs
unknown:	>
	unknown_0:
ç
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260651242
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
?
Ò
while_body_26066726
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]øH
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]øF
3while_lstm_cell_12_matmul_1_readvariableop_resource:	>øA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp×
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMulÏ
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOpÀ
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMul_1¸
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/addÈ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOpÅ
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/BiasAdd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
while/lstm_cell_12/split
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_1 
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu´
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_1©
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/add_1
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_2
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu_1¸
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
?
Ò
while_body_26066424
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]øH
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]øF
3while_lstm_cell_12_matmul_1_readvariableop_resource:	>øA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp×
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMulÏ
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOpÀ
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMul_1¸
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/addÈ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOpÅ
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/BiasAdd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
while/lstm_cell_12/split
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_1 
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu´
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_1©
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/add_1
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_2
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu_1¸
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
¹F

E__inference_lstm_12_layer_call_and_return_conditional_losses_26063431

inputs(
lstm_cell_12_26063349:	]ø(
lstm_cell_12_26063351:	>ø$
lstm_cell_12_26063353:	ø
identity¢$lstm_cell_12/StatefulPartitionedCall¢whileD
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
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2	
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
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_26063349lstm_cell_12_26063351lstm_cell_12_26063353*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260633482&
$lstm_cell_12/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_26063349lstm_cell_12_26063351lstm_cell_12_26063353*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26063362*
condR
while_cond_26063361*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2

Identity}
NoOpNoOp%^lstm_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
?
Ò
while_body_26064606
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]øH
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]øF
3while_lstm_cell_12_matmul_1_readvariableop_resource:	>øA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp×
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMulÏ
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOpÀ
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMul_1¸
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/addÈ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOpÅ
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/BiasAdd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
while/lstm_cell_12/split
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_1 
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu´
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_1©
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/add_1
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_2
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu_1¸
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
åJ
Ô

lstm_13_while_body_26066045,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:	>Q
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çK
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorL
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:	>O
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
çI
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	¢1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpÓ
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItemá
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp÷
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_13/while/lstm_cell_13/MatMulè
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpà
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_13/while/lstm_cell_13/MatMul_1Ø
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_13/while/lstm_cell_13/addà
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpå
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_13/while/lstm_cell_13/BiasAdd
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dim¯
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2"
 lstm_13/while/lstm_cell_13/split±
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2$
"lstm_13/while/lstm_cell_13/Sigmoidµ
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2&
$lstm_13/while/lstm_cell_13/Sigmoid_1Á
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2 
lstm_13/while/lstm_cell_13/mul¨
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2!
lstm_13/while/lstm_cell_13/ReluÕ
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2"
 lstm_13/while/lstm_cell_13/mul_1Ê
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2"
 lstm_13/while/lstm_cell_13/add_1µ
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2&
$lstm_13/while/lstm_cell_13/Sigmoid_2§
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2#
!lstm_13/while/lstm_cell_13/Relu_1Ù
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2"
 lstm_13/while/lstm_cell_13/mul_2
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
lstm_13/while/add/y
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
lstm_13/while/add_1/y
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity¦
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1
lstm_13/while/Identity_2Identitylstm_13/while/add:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2º
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3®
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/while/Identity_4®
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/while/Identity_5
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
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"È
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 


J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26064124

inputs

states
states_11
matmul_readvariableop_resource:	>4
 matmul_1_readvariableop_resource:
ç.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	>*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

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
A:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_namestates
ã
Í
while_cond_26067249
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26067249___redundant_placeholder06
2while_while_cond_26067249___redundant_placeholder16
2while_while_cond_26067249___redundant_placeholder26
2while_while_cond_26067249___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
×
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_26064957

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
:ÿÿÿÿÿÿÿÿÿç2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿç:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs

f
H__inference_dropout_12_layer_call_and_return_conditional_losses_26066859

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
Ø
I
-__inference_dropout_13_layer_call_fn_26067551

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
:ÿÿÿÿÿÿÿÿÿç* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260648682
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿç:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs
]
ó
(sequential_6_lstm_12_while_body_26063013F
Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counterL
Hsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations*
&sequential_6_lstm_12_while_placeholder,
(sequential_6_lstm_12_while_placeholder_1,
(sequential_6_lstm_12_while_placeholder_2,
(sequential_6_lstm_12_while_placeholder_3E
Asequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1_0
}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]ø]
Jsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øX
Isequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø'
#sequential_6_lstm_12_while_identity)
%sequential_6_lstm_12_while_identity_1)
%sequential_6_lstm_12_while_identity_2)
%sequential_6_lstm_12_while_identity_3)
%sequential_6_lstm_12_while_identity_4)
%sequential_6_lstm_12_while_identity_5C
?sequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1
{sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensorY
Fsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]ø[
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:	>øV
Gsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp¢=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp¢?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpí
Lsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2N
Lsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeÑ
>sequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0&sequential_6_lstm_12_while_placeholderUsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02@
>sequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOpHsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype02?
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp«
.sequential_6/lstm_12/while/lstm_cell_12/MatMulMatMulEsequential_6/lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø20
.sequential_6/lstm_12/while/lstm_cell_12/MatMul
?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpJsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype02A
?sequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp
0sequential_6/lstm_12/while/lstm_cell_12/MatMul_1MatMul(sequential_6_lstm_12_while_placeholder_2Gsequential_6/lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø22
0sequential_6/lstm_12/while/lstm_cell_12/MatMul_1
+sequential_6/lstm_12/while/lstm_cell_12/addAddV28sequential_6/lstm_12/while/lstm_cell_12/MatMul:product:0:sequential_6/lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2-
+sequential_6/lstm_12/while/lstm_cell_12/add
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpIsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype02@
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp
/sequential_6/lstm_12/while/lstm_cell_12/BiasAddBiasAdd/sequential_6/lstm_12/while/lstm_cell_12/add:z:0Fsequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø21
/sequential_6/lstm_12/while/lstm_cell_12/BiasAdd´
7sequential_6/lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_6/lstm_12/while/lstm_cell_12/split/split_dimß
-sequential_6/lstm_12/while/lstm_cell_12/splitSplit@sequential_6/lstm_12/while/lstm_cell_12/split/split_dim:output:08sequential_6/lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2/
-sequential_6/lstm_12/while/lstm_cell_12/split×
/sequential_6/lstm_12/while/lstm_cell_12/SigmoidSigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>21
/sequential_6/lstm_12/while/lstm_cell_12/SigmoidÛ
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>23
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1ô
+sequential_6/lstm_12/while/lstm_cell_12/mulMul5sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_1:y:0(sequential_6_lstm_12_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2-
+sequential_6/lstm_12/while/lstm_cell_12/mulÎ
,sequential_6/lstm_12/while/lstm_cell_12/ReluRelu6sequential_6/lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2.
,sequential_6/lstm_12/while/lstm_cell_12/Relu
-sequential_6/lstm_12/while/lstm_cell_12/mul_1Mul3sequential_6/lstm_12/while/lstm_cell_12/Sigmoid:y:0:sequential_6/lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2/
-sequential_6/lstm_12/while/lstm_cell_12/mul_1ý
-sequential_6/lstm_12/while/lstm_cell_12/add_1AddV2/sequential_6/lstm_12/while/lstm_cell_12/mul:z:01sequential_6/lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2/
-sequential_6/lstm_12/while/lstm_cell_12/add_1Û
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid6sequential_6/lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>23
1sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2Í
.sequential_6/lstm_12/while/lstm_cell_12/Relu_1Relu1sequential_6/lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>20
.sequential_6/lstm_12/while/lstm_cell_12/Relu_1
-sequential_6/lstm_12/while/lstm_cell_12/mul_2Mul5sequential_6/lstm_12/while/lstm_cell_12/Sigmoid_2:y:0<sequential_6/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2/
-sequential_6/lstm_12/while/lstm_cell_12/mul_2É
?sequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_6_lstm_12_while_placeholder_1&sequential_6_lstm_12_while_placeholder1sequential_6/lstm_12/while/lstm_cell_12/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItem
 sequential_6/lstm_12/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_6/lstm_12/while/add/y½
sequential_6/lstm_12/while/addAddV2&sequential_6_lstm_12_while_placeholder)sequential_6/lstm_12/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_12/while/add
"sequential_6/lstm_12/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_6/lstm_12/while/add_1/yß
 sequential_6/lstm_12/while/add_1AddV2Bsequential_6_lstm_12_while_sequential_6_lstm_12_while_loop_counter+sequential_6/lstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_12/while/add_1¿
#sequential_6/lstm_12/while/IdentityIdentity$sequential_6/lstm_12/while/add_1:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_6/lstm_12/while/Identityç
%sequential_6/lstm_12/while/Identity_1IdentityHsequential_6_lstm_12_while_sequential_6_lstm_12_while_maximum_iterations ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_1Á
%sequential_6/lstm_12/while/Identity_2Identity"sequential_6/lstm_12/while/add:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_2î
%sequential_6/lstm_12/while/Identity_3IdentityOsequential_6/lstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_6/lstm_12/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_12/while/Identity_3á
%sequential_6/lstm_12/while/Identity_4Identity1sequential_6/lstm_12/while/lstm_cell_12/mul_2:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2'
%sequential_6/lstm_12/while/Identity_4á
%sequential_6/lstm_12/while/Identity_5Identity1sequential_6/lstm_12/while/lstm_cell_12/add_1:z:0 ^sequential_6/lstm_12/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2'
%sequential_6/lstm_12/while/Identity_5Ç
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
%sequential_6_lstm_12_while_identity_5.sequential_6/lstm_12/while/Identity_5:output:0"
Gsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resourceIsequential_6_lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0"
Hsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resourceJsequential_6_lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0"
Fsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resourceHsequential_6_lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"
?sequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1Asequential_6_lstm_12_while_sequential_6_lstm_12_strided_slice_1_0"ü
{sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor}sequential_6_lstm_12_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2
>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp>sequential_6/lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp2~
=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp=sequential_6/lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp2
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
ú%
ñ
while_body_26063572
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_12_26063596_0:	]ø0
while_lstm_cell_12_26063598_0:	>ø,
while_lstm_cell_12_26063600_0:	ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_12_26063596:	]ø.
while_lstm_cell_12_26063598:	>ø*
while_lstm_cell_12_26063600:	ø¢*while/lstm_cell_12/StatefulPartitionedCallÃ
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
*while/lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_12_26063596_0while_lstm_cell_12_26063598_0while_lstm_cell_12_26063600_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260634942,
*while/lstm_cell_12/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¤
while/Identity_4Identity3while/lstm_cell_12/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_4¤
while/Identity_5Identity3while/lstm_cell_12/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_5

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
while_lstm_cell_12_26063596while_lstm_cell_12_26063596_0"<
while_lstm_cell_12_26063598while_lstm_cell_12_26063598_0"<
while_lstm_cell_12_26063600while_lstm_cell_12_26063600_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2X
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
·L
Ö
!__inference__traced_save_26067914
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
SaveV2/shape_and_slices¿
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_12_lstm_cell_12_kernel_read_readvariableop@savev2_lstm_12_lstm_cell_12_recurrent_kernel_read_readvariableop4savev2_lstm_12_lstm_cell_12_bias_read_readvariableop6savev2_lstm_13_lstm_cell_13_kernel_read_readvariableop@savev2_lstm_13_lstm_cell_13_recurrent_kernel_read_readvariableop4savev2_lstm_13_lstm_cell_13_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_m_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_m_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_m_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop=savev2_adam_lstm_12_lstm_cell_12_kernel_v_read_readvariableopGsavev2_adam_lstm_12_lstm_cell_12_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_12_lstm_cell_12_bias_v_read_readvariableop=savev2_adam_lstm_13_lstm_cell_13_kernel_v_read_readvariableopGsavev2_adam_lstm_13_lstm_cell_13_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_13_lstm_cell_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
ú: :	ç:: : : : : :	]ø:	>ø:ø:	>:
ç:: : : : :	ç::	]ø:	>ø:ø:	>:
ç::	ç::	]ø:	>ø:ø:	>:
ç:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ç: 
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
:	]ø:%	!

_output_shapes
:	>ø:!


_output_shapes	
:ø:%!

_output_shapes
:	>:&"
 
_output_shapes
:
ç:!

_output_shapes	
::
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
:	ç: 

_output_shapes
::%!

_output_shapes
:	]ø:%!

_output_shapes
:	>ø:!

_output_shapes	
:ø:%!

_output_shapes
:	>:&"
 
_output_shapes
:
ç:!

_output_shapes	
::%!

_output_shapes
:	ç: 

_output_shapes
::%!

_output_shapes
:	]ø:%!

_output_shapes
:	>ø:!

_output_shapes	
:ø:%!

_output_shapes
:	>:& "
 
_output_shapes
:
ç:!!

_output_shapes	
::"

_output_shapes
: 
Ã\
 
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067032
inputs_0>
+lstm_cell_13_matmul_readvariableop_resource:	>A
-lstm_cell_13_matmul_1_readvariableop_resource:
ç;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileF
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
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2	
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2
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
valueB"ÿÿÿÿ>   27
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
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul¼
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim÷
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu_1¡
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26066948*
condR
while_cond_26066947*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2

IdentityÈ
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
inputs/0
¶
f
-__inference_dropout_12_layer_call_fn_26066881

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
:ÿÿÿÿÿÿÿÿÿ>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260651532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
°?
Ô
while_body_26066948
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	>I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	>G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
çA
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÐ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAdd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_1¡
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Reluµ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_1ª
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Relu_1¹
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
ß
Í
while_cond_26065235
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26065235___redundant_placeholder06
2while_while_cond_26065235___redundant_placeholder16
2while_while_cond_26065235___redundant_placeholder26
2while_while_cond_26065235___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_26066725
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26066725___redundant_placeholder06
2while_while_cond_26066725___redundant_placeholder16
2while_while_cond_26066725___redundant_placeholder26
2while_while_cond_26066725___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
²
·
*__inference_lstm_12_layer_call_fn_26066843

inputs
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260646902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

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
Ç
ù
/__inference_lstm_cell_13_layer_call_fn_26067775

inputs
states_0
states_1
unknown:	>
	unknown_0:
ç
	unknown_1:	
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
<:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260639782
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

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
A:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
"
_user_specified_name
states/1
à
º
*__inference_lstm_13_layer_call_fn_26067507
inputs_0
unknown:	>
	unknown_0:
ç
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260642712
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
inputs/0
\

E__inference_lstm_13_layer_call_and_return_conditional_losses_26067485

inputs>
+lstm_cell_13_matmul_readvariableop_resource:	>A
-lstm_cell_13_matmul_1_readvariableop_resource:
ç;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileD
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
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2	
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
:ÿÿÿÿÿÿÿÿÿ>2
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
valueB"ÿÿÿÿ>   27
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
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul¼
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim÷
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu_1¡
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26067401*
condR
while_cond_26067400*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç2
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
:ÿÿÿÿÿÿÿÿÿç2

IdentityÈ
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs


J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26063978

inputs

states
states_11
matmul_readvariableop_resource:	>4
 matmul_1_readvariableop_resource:
ç.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	>*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

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
A:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_namestates
¬
ò
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065442
lstm_12_input#
lstm_12_26065420:	]ø#
lstm_12_26065422:	>ø
lstm_12_26065424:	ø#
lstm_13_26065428:	>$
lstm_13_26065430:
ç
lstm_13_26065432:	#
dense_6_26065436:	ç
dense_6_26065438:
identity¢dense_6/StatefulPartitionedCall¢lstm_12/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall´
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_26065420lstm_12_26065422lstm_12_26065424*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260646902!
lstm_12/StatefulPartitionedCall
dropout_12/PartitionedCallPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260647032
dropout_12/PartitionedCallË
lstm_13/StatefulPartitionedCallStatefulPartitionedCall#dropout_12/PartitionedCall:output:0lstm_13_26065428lstm_13_26065430lstm_13_26065432*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260648552!
lstm_13/StatefulPartitionedCall
dropout_13/PartitionedCallPartitionedCall(lstm_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260648682
dropout_13/PartitionedCall¶
dense_6/StatefulPartitionedCallStatefulPartitionedCall#dropout_13/PartitionedCall:output:0dense_6_26065436dense_6_26065438*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260649012!
dense_6/StatefulPartitionedCall
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity´
NoOpNoOp ^dense_6/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_12_input
Ð

í
lstm_12_while_cond_26065562,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1F
Blstm_12_while_lstm_12_while_cond_26065562___redundant_placeholder0F
Blstm_12_while_lstm_12_while_cond_26065562___redundant_placeholder1F
Blstm_12_while_lstm_12_while_cond_26065562___redundant_placeholder2F
Blstm_12_while_lstm_12_while_cond_26065562___redundant_placeholder3
lstm_12_while_identity

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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_26064201
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26064201___redundant_placeholder06
2while_while_cond_26064201___redundant_placeholder16
2while_while_cond_26064201___redundant_placeholder26
2while_while_cond_26064201___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
ç[

E__inference_lstm_12_layer_call_and_return_conditional_losses_26066659

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]ø@
-lstm_cell_12_matmul_1_readvariableop_resource:	>ø;
,lstm_cell_12_biasadd_readvariableop_resource:	ø
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢whileD
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
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp­
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul»
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp©
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul_1 
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/add´
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp­
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimó
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
lstm_cell_12/split
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_1
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_1
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/add_1
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu_1 
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26066575*
condR
while_cond_26066574*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>2
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
:ÿÿÿÿÿÿÿÿÿ>2

IdentityÈ
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
»
f
-__inference_dropout_13_layer_call_fn_26067556

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
:ÿÿÿÿÿÿÿÿÿç* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260649572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿç22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs


J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26067726

inputs
states_0
states_11
matmul_readvariableop_resource:	>4
 matmul_1_readvariableop_resource:
ç.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	>*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

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
A:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
"
_user_specified_name
states/1


*__inference_dense_6_layer_call_fn_26067596

inputs
unknown:	ç
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260649012
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
:ÿÿÿÿÿÿÿÿÿç: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs
ß
Í
while_cond_26066272
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26066272___redundant_placeholder06
2while_while_cond_26066272___redundant_placeholder16
2while_while_cond_26066272___redundant_placeholder26
2while_while_cond_26066272___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
ý

J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26067660

inputs
states_0
states_11
matmul_readvariableop_resource:	]ø3
 matmul_1_readvariableop_resource:	>ø.
biasadd_readvariableop_resource:	ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2	
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
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
states/1
ß
Í
while_cond_26066423
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26066423___redundant_placeholder06
2while_while_cond_26066423___redundant_placeholder16
2while_while_cond_26066423___redundant_placeholder26
2while_while_cond_26066423___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
Ô

í
lstm_13_while_cond_26066044,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1F
Blstm_13_while_lstm_13_while_cond_26066044___redundant_placeholder0F
Blstm_13_while_lstm_13_while_cond_26066044___redundant_placeholder1F
Blstm_13_while_lstm_13_while_cond_26066044___redundant_placeholder2F
Blstm_13_while_lstm_13_while_cond_26066044___redundant_placeholder3
lstm_13_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
¹
¼
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065467
lstm_12_input#
lstm_12_26065445:	]ø#
lstm_12_26065447:	>ø
lstm_12_26065449:	ø#
lstm_13_26065453:	>$
lstm_13_26065455:
ç
lstm_13_26065457:	#
dense_6_26065461:	ç
dense_6_26065463:
identity¢dense_6/StatefulPartitionedCall¢"dropout_12/StatefulPartitionedCall¢"dropout_13/StatefulPartitionedCall¢lstm_12/StatefulPartitionedCall¢lstm_13/StatefulPartitionedCall´
lstm_12/StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputlstm_12_26065445lstm_12_26065447lstm_12_26065449*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260653202!
lstm_12/StatefulPartitionedCall
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall(lstm_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_260651532$
"dropout_12/StatefulPartitionedCallÓ
lstm_13/StatefulPartitionedCallStatefulPartitionedCall+dropout_12/StatefulPartitionedCall:output:0lstm_13_26065453lstm_13_26065455lstm_13_26065457*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260651242!
lstm_13/StatefulPartitionedCallÀ
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall(lstm_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_260649572$
"dropout_13/StatefulPartitionedCall¾
dense_6/StatefulPartitionedCallStatefulPartitionedCall+dropout_13/StatefulPartitionedCall:output:0dense_6_26065461dense_6_26065463*
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
E__inference_dense_6_layer_call_and_return_conditional_losses_260649012!
dense_6/StatefulPartitionedCall
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityþ
NoOpNoOp ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall ^lstm_12/StatefulPartitionedCall ^lstm_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall2B
lstm_12/StatefulPartitionedCalllstm_12/StatefulPartitionedCall2B
lstm_13/StatefulPartitionedCalllstm_13/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_12_input


Ê
/__inference_sequential_6_layer_call_fn_26066206

inputs
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
	unknown_2:	>
	unknown_3:
ç
	unknown_4:	
	unknown_5:	ç
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_260653772
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
ý

J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26067628

inputs
states_0
states_11
matmul_readvariableop_resource:	]ø3
 matmul_1_readvariableop_resource:	>ø.
biasadd_readvariableop_resource:	ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2	
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
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
states/1
ÏJ
Ò

lstm_12_while_body_26065563,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]øP
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øK
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorL
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]øN
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:	>øI
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp¢0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp¢2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpÓ
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItemá
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp÷
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2#
!lstm_12/while/lstm_cell_12/MatMulç
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpà
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2%
#lstm_12/while/lstm_cell_12/MatMul_1Ø
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2 
lstm_12/while/lstm_cell_12/addà
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpå
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2$
"lstm_12/while/lstm_cell_12/BiasAdd
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dim«
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2"
 lstm_12/while/lstm_cell_12/split°
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2$
"lstm_12/while/lstm_cell_12/Sigmoid´
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2&
$lstm_12/while/lstm_cell_12/Sigmoid_1À
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2 
lstm_12/while/lstm_cell_12/mul§
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2!
lstm_12/while/lstm_cell_12/ReluÔ
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2"
 lstm_12/while/lstm_cell_12/mul_1É
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2"
 lstm_12/while/lstm_cell_12/add_1´
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2&
$lstm_12/while/lstm_cell_12/Sigmoid_2¦
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2#
!lstm_12/while/lstm_cell_12/Relu_1Ø
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2"
 lstm_12/while/lstm_cell_12/mul_2
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
lstm_12/while/add/y
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
lstm_12/while/add_1/y
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity¦
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2º
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3­
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0^lstm_12/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/while/Identity_4­
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0^lstm_12/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/while/Identity_5
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
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"È
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2f
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_26065039
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26065039___redundant_placeholder06
2while_while_cond_26065039___redundant_placeholder16
2while_while_cond_26065039___redundant_placeholder26
2while_while_cond_26065039___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:

f
H__inference_dropout_13_layer_call_and_return_conditional_losses_26067534

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿç:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs


J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26067758

inputs
states_0
states_11
matmul_readvariableop_resource:	>4
 matmul_1_readvariableop_resource:
ç.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	>*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

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
A:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
"
_user_specified_name
states/1
Ô

í
lstm_13_while_cond_26065710,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3.
*lstm_13_while_less_lstm_13_strided_slice_1F
Blstm_13_while_lstm_13_while_cond_26065710___redundant_placeholder0F
Blstm_13_while_lstm_13_while_cond_26065710___redundant_placeholder1F
Blstm_13_while_lstm_13_while_cond_26065710___redundant_placeholder2F
Blstm_13_while_lstm_13_while_cond_26065710___redundant_placeholder3
lstm_13_while_identity

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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
Ý
¹
*__inference_lstm_12_layer_call_fn_26066832
inputs_0
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260636412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2

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
ç³
Ï	
#__inference__wrapped_model_26063273
lstm_12_inputS
@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]øU
Bsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:	>øP
Asequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	øS
@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resource:	>V
Bsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
çP
Asequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	I
6sequential_6_dense_6_tensordot_readvariableop_resource:	çB
4sequential_6_dense_6_biasadd_readvariableop_resource:
identity¢+sequential_6/dense_6/BiasAdd/ReadVariableOp¢-sequential_6/dense_6/Tensordot/ReadVariableOp¢8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp¢7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp¢9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp¢sequential_6/lstm_12/while¢8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢sequential_6/lstm_13/whileu
sequential_6/lstm_12/ShapeShapelstm_12_input*
T0*
_output_shapes
:2
sequential_6/lstm_12/Shape
(sequential_6/lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_6/lstm_12/strided_slice/stack¢
*sequential_6/lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_12/strided_slice/stack_1¢
*sequential_6/lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_12/strided_slice/stack_2à
"sequential_6/lstm_12/strided_sliceStridedSlice#sequential_6/lstm_12/Shape:output:01sequential_6/lstm_12/strided_slice/stack:output:03sequential_6/lstm_12/strided_slice/stack_1:output:03sequential_6/lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_6/lstm_12/strided_slice
 sequential_6/lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2"
 sequential_6/lstm_12/zeros/mul/yÀ
sequential_6/lstm_12/zeros/mulMul+sequential_6/lstm_12/strided_slice:output:0)sequential_6/lstm_12/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_12/zeros/mul
!sequential_6/lstm_12/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2#
!sequential_6/lstm_12/zeros/Less/y»
sequential_6/lstm_12/zeros/LessLess"sequential_6/lstm_12/zeros/mul:z:0*sequential_6/lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_12/zeros/Less
#sequential_6/lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :>2%
#sequential_6/lstm_12/zeros/packed/1×
!sequential_6/lstm_12/zeros/packedPack+sequential_6/lstm_12/strided_slice:output:0,sequential_6/lstm_12/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_6/lstm_12/zeros/packed
 sequential_6/lstm_12/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_6/lstm_12/zeros/ConstÉ
sequential_6/lstm_12/zerosFill*sequential_6/lstm_12/zeros/packed:output:0)sequential_6/lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
sequential_6/lstm_12/zeros
"sequential_6/lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2$
"sequential_6/lstm_12/zeros_1/mul/yÆ
 sequential_6/lstm_12/zeros_1/mulMul+sequential_6/lstm_12/strided_slice:output:0+sequential_6/lstm_12/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_12/zeros_1/mul
#sequential_6/lstm_12/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#sequential_6/lstm_12/zeros_1/Less/yÃ
!sequential_6/lstm_12/zeros_1/LessLess$sequential_6/lstm_12/zeros_1/mul:z:0,sequential_6/lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_6/lstm_12/zeros_1/Less
%sequential_6/lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :>2'
%sequential_6/lstm_12/zeros_1/packed/1Ý
#sequential_6/lstm_12/zeros_1/packedPack+sequential_6/lstm_12/strided_slice:output:0.sequential_6/lstm_12/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_6/lstm_12/zeros_1/packed
"sequential_6/lstm_12/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_6/lstm_12/zeros_1/ConstÑ
sequential_6/lstm_12/zeros_1Fill,sequential_6/lstm_12/zeros_1/packed:output:0+sequential_6/lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
sequential_6/lstm_12/zeros_1
#sequential_6/lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_6/lstm_12/transpose/permÀ
sequential_6/lstm_12/transpose	Transposelstm_12_input,sequential_6/lstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2 
sequential_6/lstm_12/transpose
sequential_6/lstm_12/Shape_1Shape"sequential_6/lstm_12/transpose:y:0*
T0*
_output_shapes
:2
sequential_6/lstm_12/Shape_1¢
*sequential_6/lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_12/strided_slice_1/stack¦
,sequential_6/lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_1/stack_1¦
,sequential_6/lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_1/stack_2ì
$sequential_6/lstm_12/strided_slice_1StridedSlice%sequential_6/lstm_12/Shape_1:output:03sequential_6/lstm_12/strided_slice_1/stack:output:05sequential_6/lstm_12/strided_slice_1/stack_1:output:05sequential_6/lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_1¯
0sequential_6/lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential_6/lstm_12/TensorArrayV2/element_shape
"sequential_6/lstm_12/TensorArrayV2TensorListReserve9sequential_6/lstm_12/TensorArrayV2/element_shape:output:0-sequential_6/lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_6/lstm_12/TensorArrayV2é
Jsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2L
Jsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
<sequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_6/lstm_12/transpose:y:0Ssequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor¢
*sequential_6/lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_12/strided_slice_2/stack¦
,sequential_6/lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_2/stack_1¦
,sequential_6/lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_2/stack_2ú
$sequential_6/lstm_12/strided_slice_2StridedSlice"sequential_6/lstm_12/transpose:y:03sequential_6/lstm_12/strided_slice_2/stack:output:05sequential_6/lstm_12/strided_slice_2/stack_1:output:05sequential_6/lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_2ô
7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype029
7sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp
(sequential_6/lstm_12/lstm_cell_12/MatMulMatMul-sequential_6/lstm_12/strided_slice_2:output:0?sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2*
(sequential_6/lstm_12/lstm_cell_12/MatMulú
9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOpBsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02;
9sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpý
*sequential_6/lstm_12/lstm_cell_12/MatMul_1MatMul#sequential_6/lstm_12/zeros:output:0Asequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2,
*sequential_6/lstm_12/lstm_cell_12/MatMul_1ô
%sequential_6/lstm_12/lstm_cell_12/addAddV22sequential_6/lstm_12/lstm_cell_12/MatMul:product:04sequential_6/lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2'
%sequential_6/lstm_12/lstm_cell_12/addó
8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOpAsequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02:
8sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp
)sequential_6/lstm_12/lstm_cell_12/BiasAddBiasAdd)sequential_6/lstm_12/lstm_cell_12/add:z:0@sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2+
)sequential_6/lstm_12/lstm_cell_12/BiasAdd¨
1sequential_6/lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_6/lstm_12/lstm_cell_12/split/split_dimÇ
'sequential_6/lstm_12/lstm_cell_12/splitSplit:sequential_6/lstm_12/lstm_cell_12/split/split_dim:output:02sequential_6/lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2)
'sequential_6/lstm_12/lstm_cell_12/splitÅ
)sequential_6/lstm_12/lstm_cell_12/SigmoidSigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2+
)sequential_6/lstm_12/lstm_cell_12/SigmoidÉ
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_1Sigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2-
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_1ß
%sequential_6/lstm_12/lstm_cell_12/mulMul/sequential_6/lstm_12/lstm_cell_12/Sigmoid_1:y:0%sequential_6/lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2'
%sequential_6/lstm_12/lstm_cell_12/mul¼
&sequential_6/lstm_12/lstm_cell_12/ReluRelu0sequential_6/lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2(
&sequential_6/lstm_12/lstm_cell_12/Reluð
'sequential_6/lstm_12/lstm_cell_12/mul_1Mul-sequential_6/lstm_12/lstm_cell_12/Sigmoid:y:04sequential_6/lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2)
'sequential_6/lstm_12/lstm_cell_12/mul_1å
'sequential_6/lstm_12/lstm_cell_12/add_1AddV2)sequential_6/lstm_12/lstm_cell_12/mul:z:0+sequential_6/lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2)
'sequential_6/lstm_12/lstm_cell_12/add_1É
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_2Sigmoid0sequential_6/lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2-
+sequential_6/lstm_12/lstm_cell_12/Sigmoid_2»
(sequential_6/lstm_12/lstm_cell_12/Relu_1Relu+sequential_6/lstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2*
(sequential_6/lstm_12/lstm_cell_12/Relu_1ô
'sequential_6/lstm_12/lstm_cell_12/mul_2Mul/sequential_6/lstm_12/lstm_cell_12/Sigmoid_2:y:06sequential_6/lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2)
'sequential_6/lstm_12/lstm_cell_12/mul_2¹
2sequential_6/lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   24
2sequential_6/lstm_12/TensorArrayV2_1/element_shape
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
sequential_6/lstm_12/time©
-sequential_6/lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential_6/lstm_12/while/maximum_iterations
'sequential_6/lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/lstm_12/while/loop_counterÊ
sequential_6/lstm_12/whileWhile0sequential_6/lstm_12/while/loop_counter:output:06sequential_6/lstm_12/while/maximum_iterations:output:0"sequential_6/lstm_12/time:output:0-sequential_6/lstm_12/TensorArrayV2_1:handle:0#sequential_6/lstm_12/zeros:output:0%sequential_6/lstm_12/zeros_1:output:0-sequential_6/lstm_12/strided_slice_1:output:0Lsequential_6/lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_6_lstm_12_lstm_cell_12_matmul_readvariableop_resourceBsequential_6_lstm_12_lstm_cell_12_matmul_1_readvariableop_resourceAsequential_6_lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_6_lstm_12_while_body_26063013*4
cond,R*
(sequential_6_lstm_12_while_cond_26063012*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
sequential_6/lstm_12/whileß
Esequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2G
Esequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape¼
7sequential_6/lstm_12/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_6/lstm_12/while:output:3Nsequential_6/lstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype029
7sequential_6/lstm_12/TensorArrayV2Stack/TensorListStack«
*sequential_6/lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2,
*sequential_6/lstm_12/strided_slice_3/stack¦
,sequential_6/lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_6/lstm_12/strided_slice_3/stack_1¦
,sequential_6/lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_12/strided_slice_3/stack_2
$sequential_6/lstm_12/strided_slice_3StridedSlice@sequential_6/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/lstm_12/strided_slice_3/stack:output:05sequential_6/lstm_12/strided_slice_3/stack_1:output:05sequential_6/lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2&
$sequential_6/lstm_12/strided_slice_3£
%sequential_6/lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_6/lstm_12/transpose_1/permù
 sequential_6/lstm_12/transpose_1	Transpose@sequential_6/lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_6/lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2"
 sequential_6/lstm_12/transpose_1
sequential_6/lstm_12/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_6/lstm_12/runtime¬
 sequential_6/dropout_12/IdentityIdentity$sequential_6/lstm_12/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2"
 sequential_6/dropout_12/Identity
sequential_6/lstm_13/ShapeShape)sequential_6/dropout_12/Identity:output:0*
T0*
_output_shapes
:2
sequential_6/lstm_13/Shape
(sequential_6/lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(sequential_6/lstm_13/strided_slice/stack¢
*sequential_6/lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_13/strided_slice/stack_1¢
*sequential_6/lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*sequential_6/lstm_13/strided_slice/stack_2à
"sequential_6/lstm_13/strided_sliceStridedSlice#sequential_6/lstm_13/Shape:output:01sequential_6/lstm_13/strided_slice/stack:output:03sequential_6/lstm_13/strided_slice/stack_1:output:03sequential_6/lstm_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"sequential_6/lstm_13/strided_slice
 sequential_6/lstm_13/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2"
 sequential_6/lstm_13/zeros/mul/yÀ
sequential_6/lstm_13/zeros/mulMul+sequential_6/lstm_13/strided_slice:output:0)sequential_6/lstm_13/zeros/mul/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_13/zeros/mul
!sequential_6/lstm_13/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2#
!sequential_6/lstm_13/zeros/Less/y»
sequential_6/lstm_13/zeros/LessLess"sequential_6/lstm_13/zeros/mul:z:0*sequential_6/lstm_13/zeros/Less/y:output:0*
T0*
_output_shapes
: 2!
sequential_6/lstm_13/zeros/Less
#sequential_6/lstm_13/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ç2%
#sequential_6/lstm_13/zeros/packed/1×
!sequential_6/lstm_13/zeros/packedPack+sequential_6/lstm_13/strided_slice:output:0,sequential_6/lstm_13/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2#
!sequential_6/lstm_13/zeros/packed
 sequential_6/lstm_13/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 sequential_6/lstm_13/zeros/ConstÊ
sequential_6/lstm_13/zerosFill*sequential_6/lstm_13/zeros/packed:output:0)sequential_6/lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
sequential_6/lstm_13/zeros
"sequential_6/lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2$
"sequential_6/lstm_13/zeros_1/mul/yÆ
 sequential_6/lstm_13/zeros_1/mulMul+sequential_6/lstm_13/strided_slice:output:0+sequential_6/lstm_13/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_13/zeros_1/mul
#sequential_6/lstm_13/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2%
#sequential_6/lstm_13/zeros_1/Less/yÃ
!sequential_6/lstm_13/zeros_1/LessLess$sequential_6/lstm_13/zeros_1/mul:z:0,sequential_6/lstm_13/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2#
!sequential_6/lstm_13/zeros_1/Less
%sequential_6/lstm_13/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :ç2'
%sequential_6/lstm_13/zeros_1/packed/1Ý
#sequential_6/lstm_13/zeros_1/packedPack+sequential_6/lstm_13/strided_slice:output:0.sequential_6/lstm_13/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2%
#sequential_6/lstm_13/zeros_1/packed
"sequential_6/lstm_13/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"sequential_6/lstm_13/zeros_1/ConstÒ
sequential_6/lstm_13/zeros_1Fill,sequential_6/lstm_13/zeros_1/packed:output:0+sequential_6/lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
sequential_6/lstm_13/zeros_1
#sequential_6/lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2%
#sequential_6/lstm_13/transpose/permÜ
sequential_6/lstm_13/transpose	Transpose)sequential_6/dropout_12/Identity:output:0,sequential_6/lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2 
sequential_6/lstm_13/transpose
sequential_6/lstm_13/Shape_1Shape"sequential_6/lstm_13/transpose:y:0*
T0*
_output_shapes
:2
sequential_6/lstm_13/Shape_1¢
*sequential_6/lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_13/strided_slice_1/stack¦
,sequential_6/lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_1/stack_1¦
,sequential_6/lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_1/stack_2ì
$sequential_6/lstm_13/strided_slice_1StridedSlice%sequential_6/lstm_13/Shape_1:output:03sequential_6/lstm_13/strided_slice_1/stack:output:05sequential_6/lstm_13/strided_slice_1/stack_1:output:05sequential_6/lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_1¯
0sequential_6/lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0sequential_6/lstm_13/TensorArrayV2/element_shape
"sequential_6/lstm_13/TensorArrayV2TensorListReserve9sequential_6/lstm_13/TensorArrayV2/element_shape:output:0-sequential_6/lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02$
"sequential_6/lstm_13/TensorArrayV2é
Jsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2L
Jsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeÌ
<sequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor"sequential_6/lstm_13/transpose:y:0Ssequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02>
<sequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor¢
*sequential_6/lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*sequential_6/lstm_13/strided_slice_2/stack¦
,sequential_6/lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_2/stack_1¦
,sequential_6/lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_2/stack_2ú
$sequential_6/lstm_13/strided_slice_2StridedSlice"sequential_6/lstm_13/transpose:y:03sequential_6/lstm_13/strided_slice_2/stack:output:05sequential_6/lstm_13/strided_slice_2/stack_1:output:05sequential_6/lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_2ô
7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype029
7sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp
(sequential_6/lstm_13/lstm_cell_13/MatMulMatMul-sequential_6/lstm_13/strided_slice_2:output:0?sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_6/lstm_13/lstm_cell_13/MatMulû
9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpBsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02;
9sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpý
*sequential_6/lstm_13/lstm_cell_13/MatMul_1MatMul#sequential_6/lstm_13/zeros:output:0Asequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*sequential_6/lstm_13/lstm_cell_13/MatMul_1ô
%sequential_6/lstm_13/lstm_cell_13/addAddV22sequential_6/lstm_13/lstm_cell_13/MatMul:product:04sequential_6/lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_6/lstm_13/lstm_cell_13/addó
8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpAsequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02:
8sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp
)sequential_6/lstm_13/lstm_cell_13/BiasAddBiasAdd)sequential_6/lstm_13/lstm_cell_13/add:z:0@sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_6/lstm_13/lstm_cell_13/BiasAdd¨
1sequential_6/lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :23
1sequential_6/lstm_13/lstm_cell_13/split/split_dimË
'sequential_6/lstm_13/lstm_cell_13/splitSplit:sequential_6/lstm_13/lstm_cell_13/split/split_dim:output:02sequential_6/lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2)
'sequential_6/lstm_13/lstm_cell_13/splitÆ
)sequential_6/lstm_13/lstm_cell_13/SigmoidSigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2+
)sequential_6/lstm_13/lstm_cell_13/SigmoidÊ
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_1Sigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2-
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_1à
%sequential_6/lstm_13/lstm_cell_13/mulMul/sequential_6/lstm_13/lstm_cell_13/Sigmoid_1:y:0%sequential_6/lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2'
%sequential_6/lstm_13/lstm_cell_13/mul½
&sequential_6/lstm_13/lstm_cell_13/ReluRelu0sequential_6/lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2(
&sequential_6/lstm_13/lstm_cell_13/Reluñ
'sequential_6/lstm_13/lstm_cell_13/mul_1Mul-sequential_6/lstm_13/lstm_cell_13/Sigmoid:y:04sequential_6/lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2)
'sequential_6/lstm_13/lstm_cell_13/mul_1æ
'sequential_6/lstm_13/lstm_cell_13/add_1AddV2)sequential_6/lstm_13/lstm_cell_13/mul:z:0+sequential_6/lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2)
'sequential_6/lstm_13/lstm_cell_13/add_1Ê
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_2Sigmoid0sequential_6/lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2-
+sequential_6/lstm_13/lstm_cell_13/Sigmoid_2¼
(sequential_6/lstm_13/lstm_cell_13/Relu_1Relu+sequential_6/lstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2*
(sequential_6/lstm_13/lstm_cell_13/Relu_1õ
'sequential_6/lstm_13/lstm_cell_13/mul_2Mul/sequential_6/lstm_13/lstm_cell_13/Sigmoid_2:y:06sequential_6/lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2)
'sequential_6/lstm_13/lstm_cell_13/mul_2¹
2sequential_6/lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   24
2sequential_6/lstm_13/TensorArrayV2_1/element_shape
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
sequential_6/lstm_13/time©
-sequential_6/lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2/
-sequential_6/lstm_13/while/maximum_iterations
'sequential_6/lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2)
'sequential_6/lstm_13/while/loop_counterÎ
sequential_6/lstm_13/whileWhile0sequential_6/lstm_13/while/loop_counter:output:06sequential_6/lstm_13/while/maximum_iterations:output:0"sequential_6/lstm_13/time:output:0-sequential_6/lstm_13/TensorArrayV2_1:handle:0#sequential_6/lstm_13/zeros:output:0%sequential_6/lstm_13/zeros_1:output:0-sequential_6/lstm_13/strided_slice_1:output:0Lsequential_6/lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_6_lstm_13_lstm_cell_13_matmul_readvariableop_resourceBsequential_6_lstm_13_lstm_cell_13_matmul_1_readvariableop_resourceAsequential_6_lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *4
body,R*
(sequential_6_lstm_13_while_body_26063161*4
cond,R*
(sequential_6_lstm_13_while_cond_26063160*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
sequential_6/lstm_13/whileß
Esequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2G
Esequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape½
7sequential_6/lstm_13/TensorArrayV2Stack/TensorListStackTensorListStack#sequential_6/lstm_13/while:output:3Nsequential_6/lstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
element_dtype029
7sequential_6/lstm_13/TensorArrayV2Stack/TensorListStack«
*sequential_6/lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2,
*sequential_6/lstm_13/strided_slice_3/stack¦
,sequential_6/lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,sequential_6/lstm_13/strided_slice_3/stack_1¦
,sequential_6/lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,sequential_6/lstm_13/strided_slice_3/stack_2
$sequential_6/lstm_13/strided_slice_3StridedSlice@sequential_6/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:03sequential_6/lstm_13/strided_slice_3/stack:output:05sequential_6/lstm_13/strided_slice_3/stack_1:output:05sequential_6/lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
shrink_axis_mask2&
$sequential_6/lstm_13/strided_slice_3£
%sequential_6/lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2'
%sequential_6/lstm_13/transpose_1/permú
 sequential_6/lstm_13/transpose_1	Transpose@sequential_6/lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0.sequential_6/lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2"
 sequential_6/lstm_13/transpose_1
sequential_6/lstm_13/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_6/lstm_13/runtime­
 sequential_6/dropout_13/IdentityIdentity$sequential_6/lstm_13/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2"
 sequential_6/dropout_13/IdentityÖ
-sequential_6/dense_6/Tensordot/ReadVariableOpReadVariableOp6sequential_6_dense_6_tensordot_readvariableop_resource*
_output_shapes
:	ç*
dtype02/
-sequential_6/dense_6/Tensordot/ReadVariableOp
#sequential_6/dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_6/dense_6/Tensordot/axes
#sequential_6/dense_6/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_6/dense_6/Tensordot/free¥
$sequential_6/dense_6/Tensordot/ShapeShape)sequential_6/dropout_13/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_6/dense_6/Tensordot/Shape
,sequential_6/dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/dense_6/Tensordot/GatherV2/axisº
'sequential_6/dense_6/Tensordot/GatherV2GatherV2-sequential_6/dense_6/Tensordot/Shape:output:0,sequential_6/dense_6/Tensordot/free:output:05sequential_6/dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_6/dense_6/Tensordot/GatherV2¢
.sequential_6/dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_6/dense_6/Tensordot/GatherV2_1/axisÀ
)sequential_6/dense_6/Tensordot/GatherV2_1GatherV2-sequential_6/dense_6/Tensordot/Shape:output:0,sequential_6/dense_6/Tensordot/axes:output:07sequential_6/dense_6/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_6/dense_6/Tensordot/GatherV2_1
$sequential_6/dense_6/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_6/dense_6/Tensordot/ConstÔ
#sequential_6/dense_6/Tensordot/ProdProd0sequential_6/dense_6/Tensordot/GatherV2:output:0-sequential_6/dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_6/dense_6/Tensordot/Prod
&sequential_6/dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_6/dense_6/Tensordot/Const_1Ü
%sequential_6/dense_6/Tensordot/Prod_1Prod2sequential_6/dense_6/Tensordot/GatherV2_1:output:0/sequential_6/dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_6/dense_6/Tensordot/Prod_1
*sequential_6/dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_6/dense_6/Tensordot/concat/axis
%sequential_6/dense_6/Tensordot/concatConcatV2,sequential_6/dense_6/Tensordot/free:output:0,sequential_6/dense_6/Tensordot/axes:output:03sequential_6/dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_6/dense_6/Tensordot/concatà
$sequential_6/dense_6/Tensordot/stackPack,sequential_6/dense_6/Tensordot/Prod:output:0.sequential_6/dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_6/dense_6/Tensordot/stackó
(sequential_6/dense_6/Tensordot/transpose	Transpose)sequential_6/dropout_13/Identity:output:0.sequential_6/dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2*
(sequential_6/dense_6/Tensordot/transposeó
&sequential_6/dense_6/Tensordot/ReshapeReshape,sequential_6/dense_6/Tensordot/transpose:y:0-sequential_6/dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_6/dense_6/Tensordot/Reshapeò
%sequential_6/dense_6/Tensordot/MatMulMatMul/sequential_6/dense_6/Tensordot/Reshape:output:05sequential_6/dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_6/dense_6/Tensordot/MatMul
&sequential_6/dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_6/dense_6/Tensordot/Const_2
,sequential_6/dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_6/dense_6/Tensordot/concat_1/axis¦
'sequential_6/dense_6/Tensordot/concat_1ConcatV20sequential_6/dense_6/Tensordot/GatherV2:output:0/sequential_6/dense_6/Tensordot/Const_2:output:05sequential_6/dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_6/dense_6/Tensordot/concat_1ä
sequential_6/dense_6/TensordotReshape/sequential_6/dense_6/Tensordot/MatMul:product:00sequential_6/dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_6/dense_6/TensordotË
+sequential_6/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_6_dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_6/dense_6/BiasAdd/ReadVariableOpÛ
sequential_6/dense_6/BiasAddBiasAdd'sequential_6/dense_6/Tensordot:output:03sequential_6/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense_6/BiasAdd¤
sequential_6/dense_6/SoftmaxSoftmax%sequential_6/dense_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_6/dense_6/Softmax
IdentityIdentity&sequential_6/dense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÈ
NoOpNoOp,^sequential_6/dense_6/BiasAdd/ReadVariableOp.^sequential_6/dense_6/Tensordot/ReadVariableOp9^sequential_6/lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp8^sequential_6/lstm_12/lstm_cell_12/MatMul/ReadVariableOp:^sequential_6/lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^sequential_6/lstm_12/while9^sequential_6/lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp8^sequential_6/lstm_13/lstm_cell_13/MatMul/ReadVariableOp:^sequential_6/lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^sequential_6/lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2Z
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
:ÿÿÿÿÿÿÿÿÿ]
'
_user_specified_namelstm_12_input
²
·
*__inference_lstm_12_layer_call_fn_26066854

inputs
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_12_layer_call_and_return_conditional_losses_260653202
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

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
?
Ò
while_body_26066273
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]øH
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]øF
3while_lstm_cell_12_matmul_1_readvariableop_resource:	>øA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp×
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMulÏ
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOpÀ
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMul_1¸
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/addÈ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOpÅ
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/BiasAdd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
while/lstm_cell_12/split
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_1 
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu´
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_1©
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/add_1
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_2
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu_1¸
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
à
º
*__inference_lstm_13_layer_call_fn_26067496
inputs_0
unknown:	>
	unknown_0:
ç
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_lstm_13_layer_call_and_return_conditional_losses_260640612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
inputs/0
Û
ñ
(sequential_6_lstm_13_while_cond_26063160F
Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counterL
Hsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations*
&sequential_6_lstm_13_while_placeholder,
(sequential_6_lstm_13_while_placeholder_1,
(sequential_6_lstm_13_while_placeholder_2,
(sequential_6_lstm_13_while_placeholder_3H
Dsequential_6_lstm_13_while_less_sequential_6_lstm_13_strided_slice_1`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26063160___redundant_placeholder0`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26063160___redundant_placeholder1`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26063160___redundant_placeholder2`
\sequential_6_lstm_13_while_sequential_6_lstm_13_while_cond_26063160___redundant_placeholder3'
#sequential_6_lstm_13_while_identity
Ù
sequential_6/lstm_13/while/LessLess&sequential_6_lstm_13_while_placeholderDsequential_6_lstm_13_while_less_sequential_6_lstm_13_strided_slice_1*
T0*
_output_shapes
: 2!
sequential_6/lstm_13/while/Less
#sequential_6/lstm_13/while/IdentityIdentity#sequential_6/lstm_13/while/Less:z:0*
T0
*
_output_shapes
: 2%
#sequential_6/lstm_13/while/Identity"S
#sequential_6_lstm_13_while_identity,sequential_6/lstm_13/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
õ

J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26063348

inputs

states
states_11
matmul_readvariableop_resource:	]ø3
 matmul_1_readvariableop_resource:	>ø.
biasadd_readvariableop_resource:	ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2	
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
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_namestates
º
ø
/__inference_lstm_cell_12_layer_call_fn_26067677

inputs
states_0
states_1
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
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
9:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260633482
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
"
_user_specified_name
states/1
ã
Í
while_cond_26067400
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26067400___redundant_placeholder06
2while_while_cond_26067400___redundant_placeholder16
2while_while_cond_26067400___redundant_placeholder26
2while_while_cond_26067400___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
ç[

E__inference_lstm_12_layer_call_and_return_conditional_losses_26065320

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]ø@
-lstm_cell_12_matmul_1_readvariableop_resource:	>ø;
,lstm_cell_12_biasadd_readvariableop_resource:	ø
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢whileD
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
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp­
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul»
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp©
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul_1 
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/add´
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp­
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimó
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
lstm_cell_12/split
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_1
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_1
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/add_1
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu_1 
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26065236*
condR
while_cond_26065235*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>2
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
:ÿÿÿÿÿÿÿÿÿ>2

IdentityÈ
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
×
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_26067546

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
:ÿÿÿÿÿÿÿÿÿç2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿç:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs
ï

J__inference_sequential_6_layer_call_and_return_conditional_losses_26066164

inputsF
3lstm_12_lstm_cell_12_matmul_readvariableop_resource:	]øH
5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource:	>øC
4lstm_12_lstm_cell_12_biasadd_readvariableop_resource:	øF
3lstm_13_lstm_cell_13_matmul_readvariableop_resource:	>I
5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource:
çC
4lstm_13_lstm_cell_13_biasadd_readvariableop_resource:	<
)dense_6_tensordot_readvariableop_resource:	ç5
'dense_6_biasadd_readvariableop_resource:
identity¢dense_6/BiasAdd/ReadVariableOp¢ dense_6/Tensordot/ReadVariableOp¢+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp¢*lstm_12/lstm_cell_12/MatMul/ReadVariableOp¢,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp¢lstm_12/while¢+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp¢*lstm_13/lstm_cell_13/MatMul/ReadVariableOp¢,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp¢lstm_13/whileT
lstm_12/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_12/Shape
lstm_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice/stack
lstm_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_1
lstm_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_12/strided_slice/stack_2
lstm_12/strided_sliceStridedSlicelstm_12/Shape:output:0$lstm_12/strided_slice/stack:output:0&lstm_12/strided_slice/stack_1:output:0&lstm_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slicel
lstm_12/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
lstm_12/zeros/mul/y
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
B :è2
lstm_12/zeros/Less/y
lstm_12/zeros/LessLesslstm_12/zeros/mul:z:0lstm_12/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros/Lessr
lstm_12/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :>2
lstm_12/zeros/packed/1£
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
lstm_12/zeros/Const
lstm_12/zerosFilllstm_12/zeros/packed:output:0lstm_12/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/zerosp
lstm_12/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
lstm_12/zeros_1/mul/y
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
B :è2
lstm_12/zeros_1/Less/y
lstm_12/zeros_1/LessLesslstm_12/zeros_1/mul:z:0lstm_12/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_12/zeros_1/Lessv
lstm_12/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :>2
lstm_12/zeros_1/packed/1©
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
lstm_12/zeros_1/Const
lstm_12/zeros_1Filllstm_12/zeros_1/packed:output:0lstm_12/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/zeros_1
lstm_12/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose/perm
lstm_12/transpose	Transposeinputslstm_12/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_12/transposeg
lstm_12/Shape_1Shapelstm_12/transpose:y:0*
T0*
_output_shapes
:2
lstm_12/Shape_1
lstm_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_1/stack
lstm_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_1
lstm_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_1/stack_2
lstm_12/strided_slice_1StridedSlicelstm_12/Shape_1:output:0&lstm_12/strided_slice_1/stack:output:0(lstm_12/strided_slice_1/stack_1:output:0(lstm_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_12/strided_slice_1
#lstm_12/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_12/TensorArrayV2/element_shapeÒ
lstm_12/TensorArrayV2TensorListReserve,lstm_12/TensorArrayV2/element_shape:output:0 lstm_12/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_12/TensorArrayV2Ï
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2?
=lstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_12/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_12/transpose:y:0Flstm_12/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_12/TensorArrayUnstack/TensorListFromTensor
lstm_12/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_12/strided_slice_2/stack
lstm_12/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_1
lstm_12/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_2/stack_2¬
lstm_12/strided_slice_2StridedSlicelstm_12/transpose:y:0&lstm_12/strided_slice_2/stack:output:0(lstm_12/strided_slice_2/stack_1:output:0(lstm_12/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_12/strided_slice_2Í
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3lstm_12_lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02,
*lstm_12/lstm_cell_12/MatMul/ReadVariableOpÍ
lstm_12/lstm_cell_12/MatMulMatMul lstm_12/strided_slice_2:output:02lstm_12/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_12/lstm_cell_12/MatMulÓ
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02.
,lstm_12/lstm_cell_12/MatMul_1/ReadVariableOpÉ
lstm_12/lstm_cell_12/MatMul_1MatMullstm_12/zeros:output:04lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_12/lstm_cell_12/MatMul_1À
lstm_12/lstm_cell_12/addAddV2%lstm_12/lstm_cell_12/MatMul:product:0'lstm_12/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_12/lstm_cell_12/addÌ
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02-
+lstm_12/lstm_cell_12/BiasAdd/ReadVariableOpÍ
lstm_12/lstm_cell_12/BiasAddBiasAddlstm_12/lstm_cell_12/add:z:03lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_12/lstm_cell_12/BiasAdd
$lstm_12/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_12/lstm_cell_12/split/split_dim
lstm_12/lstm_cell_12/splitSplit-lstm_12/lstm_cell_12/split/split_dim:output:0%lstm_12/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
lstm_12/lstm_cell_12/split
lstm_12/lstm_cell_12/SigmoidSigmoid#lstm_12/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/Sigmoid¢
lstm_12/lstm_cell_12/Sigmoid_1Sigmoid#lstm_12/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2 
lstm_12/lstm_cell_12/Sigmoid_1«
lstm_12/lstm_cell_12/mulMul"lstm_12/lstm_cell_12/Sigmoid_1:y:0lstm_12/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/mul
lstm_12/lstm_cell_12/ReluRelu#lstm_12/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/Relu¼
lstm_12/lstm_cell_12/mul_1Mul lstm_12/lstm_cell_12/Sigmoid:y:0'lstm_12/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/mul_1±
lstm_12/lstm_cell_12/add_1AddV2lstm_12/lstm_cell_12/mul:z:0lstm_12/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/add_1¢
lstm_12/lstm_cell_12/Sigmoid_2Sigmoid#lstm_12/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2 
lstm_12/lstm_cell_12/Sigmoid_2
lstm_12/lstm_cell_12/Relu_1Relulstm_12/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/Relu_1À
lstm_12/lstm_cell_12/mul_2Mul"lstm_12/lstm_cell_12/Sigmoid_2:y:0)lstm_12/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/lstm_cell_12/mul_2
%lstm_12/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2'
%lstm_12/TensorArrayV2_1/element_shapeØ
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
lstm_12/time
 lstm_12/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_12/while/maximum_iterationsz
lstm_12/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_12/while/loop_counter
lstm_12/whileWhile#lstm_12/while/loop_counter:output:0)lstm_12/while/maximum_iterations:output:0lstm_12/time:output:0 lstm_12/TensorArrayV2_1:handle:0lstm_12/zeros:output:0lstm_12/zeros_1:output:0 lstm_12/strided_slice_1:output:0?lstm_12/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_12_lstm_cell_12_matmul_readvariableop_resource5lstm_12_lstm_cell_12_matmul_1_readvariableop_resource4lstm_12_lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_12_while_body_26065890*'
condR
lstm_12_while_cond_26065889*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
lstm_12/whileÅ
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2:
8lstm_12/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_12/TensorArrayV2Stack/TensorListStackTensorListStacklstm_12/while:output:3Alstm_12/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02,
*lstm_12/TensorArrayV2Stack/TensorListStack
lstm_12/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_12/strided_slice_3/stack
lstm_12/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_12/strided_slice_3/stack_1
lstm_12/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_12/strided_slice_3/stack_2Ê
lstm_12/strided_slice_3StridedSlice3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_12/strided_slice_3/stack:output:0(lstm_12/strided_slice_3/stack_1:output:0(lstm_12/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
lstm_12/strided_slice_3
lstm_12/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_12/transpose_1/permÅ
lstm_12/transpose_1	Transpose3lstm_12/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_12/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
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
 *«ªª?2
dropout_12/dropout/Const©
dropout_12/dropout/MulMullstm_12/transpose_1:y:0!dropout_12/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
dropout_12/dropout/Mul{
dropout_12/dropout/ShapeShapelstm_12/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_12/dropout/ShapeÙ
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
dtype021
/dropout_12/dropout/random_uniform/RandomUniform
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2#
!dropout_12/dropout/GreaterEqual/yî
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2!
dropout_12/dropout/GreaterEqual¤
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
dropout_12/dropout/Castª
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
dropout_12/dropout/Mul_1j
lstm_13/ShapeShapedropout_12/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_13/Shape
lstm_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice/stack
lstm_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_1
lstm_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_13/strided_slice/stack_2
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
B :ç2
lstm_13/zeros/mul/y
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
B :è2
lstm_13/zeros/Less/y
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
B :ç2
lstm_13/zeros/packed/1£
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
lstm_13/zeros/Const
lstm_13/zerosFilllstm_13/zeros/packed:output:0lstm_13/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/zerosq
lstm_13/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
lstm_13/zeros_1/mul/y
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
B :è2
lstm_13/zeros_1/Less/y
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
B :ç2
lstm_13/zeros_1/packed/1©
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
lstm_13/zeros_1/Const
lstm_13/zeros_1Filllstm_13/zeros_1/packed:output:0lstm_13/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/zeros_1
lstm_13/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose/perm¨
lstm_13/transpose	Transposedropout_12/dropout/Mul_1:z:0lstm_13/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_13/transposeg
lstm_13/Shape_1Shapelstm_13/transpose:y:0*
T0*
_output_shapes
:2
lstm_13/Shape_1
lstm_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_1/stack
lstm_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_1
lstm_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_1/stack_2
lstm_13/strided_slice_1StridedSlicelstm_13/Shape_1:output:0&lstm_13/strided_slice_1/stack:output:0(lstm_13/strided_slice_1/stack_1:output:0(lstm_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_13/strided_slice_1
#lstm_13/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#lstm_13/TensorArrayV2/element_shapeÒ
lstm_13/TensorArrayV2TensorListReserve,lstm_13/TensorArrayV2/element_shape:output:0 lstm_13/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_13/TensorArrayV2Ï
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2?
=lstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape
/lstm_13/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_13/transpose:y:0Flstm_13/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type021
/lstm_13/TensorArrayUnstack/TensorListFromTensor
lstm_13/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_13/strided_slice_2/stack
lstm_13/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_1
lstm_13/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_2/stack_2¬
lstm_13/strided_slice_2StridedSlicelstm_13/transpose:y:0&lstm_13/strided_slice_2/stack:output:0(lstm_13/strided_slice_2/stack_1:output:0(lstm_13/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
lstm_13/strided_slice_2Í
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3lstm_13_lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype02,
*lstm_13/lstm_cell_13/MatMul/ReadVariableOpÍ
lstm_13/lstm_cell_13/MatMulMatMul lstm_13/strided_slice_2:output:02lstm_13/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/MatMulÔ
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02.
,lstm_13/lstm_cell_13/MatMul_1/ReadVariableOpÉ
lstm_13/lstm_cell_13/MatMul_1MatMullstm_13/zeros:output:04lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/MatMul_1À
lstm_13/lstm_cell_13/addAddV2%lstm_13/lstm_cell_13/MatMul:product:0'lstm_13/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/addÌ
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+lstm_13/lstm_cell_13/BiasAdd/ReadVariableOpÍ
lstm_13/lstm_cell_13/BiasAddBiasAddlstm_13/lstm_cell_13/add:z:03lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_13/lstm_cell_13/BiasAdd
$lstm_13/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm_13/lstm_cell_13/split/split_dim
lstm_13/lstm_cell_13/splitSplit-lstm_13/lstm_cell_13/split/split_dim:output:0%lstm_13/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
lstm_13/lstm_cell_13/split
lstm_13/lstm_cell_13/SigmoidSigmoid#lstm_13/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/Sigmoid£
lstm_13/lstm_cell_13/Sigmoid_1Sigmoid#lstm_13/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2 
lstm_13/lstm_cell_13/Sigmoid_1¬
lstm_13/lstm_cell_13/mulMul"lstm_13/lstm_cell_13/Sigmoid_1:y:0lstm_13/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/mul
lstm_13/lstm_cell_13/ReluRelu#lstm_13/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/Relu½
lstm_13/lstm_cell_13/mul_1Mul lstm_13/lstm_cell_13/Sigmoid:y:0'lstm_13/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/mul_1²
lstm_13/lstm_cell_13/add_1AddV2lstm_13/lstm_cell_13/mul:z:0lstm_13/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/add_1£
lstm_13/lstm_cell_13/Sigmoid_2Sigmoid#lstm_13/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2 
lstm_13/lstm_cell_13/Sigmoid_2
lstm_13/lstm_cell_13/Relu_1Relulstm_13/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/Relu_1Á
lstm_13/lstm_cell_13/mul_2Mul"lstm_13/lstm_cell_13/Sigmoid_2:y:0)lstm_13/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/lstm_cell_13/mul_2
%lstm_13/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2'
%lstm_13/TensorArrayV2_1/element_shapeØ
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
lstm_13/time
 lstm_13/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm_13/while/maximum_iterationsz
lstm_13/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_13/while/loop_counter
lstm_13/whileWhile#lstm_13/while/loop_counter:output:0)lstm_13/while/maximum_iterations:output:0lstm_13/time:output:0 lstm_13/TensorArrayV2_1:handle:0lstm_13/zeros:output:0lstm_13/zeros_1:output:0 lstm_13/strided_slice_1:output:0?lstm_13/TensorArrayUnstack/TensorListFromTensor:output_handle:03lstm_13_lstm_cell_13_matmul_readvariableop_resource5lstm_13_lstm_cell_13_matmul_1_readvariableop_resource4lstm_13_lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *'
bodyR
lstm_13_while_body_26066045*'
condR
lstm_13_while_cond_26066044*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
lstm_13/whileÅ
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2:
8lstm_13/TensorArrayV2Stack/TensorListStack/element_shape
*lstm_13/TensorArrayV2Stack/TensorListStackTensorListStacklstm_13/while:output:3Alstm_13/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
element_dtype02,
*lstm_13/TensorArrayV2Stack/TensorListStack
lstm_13/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_13/strided_slice_3/stack
lstm_13/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
lstm_13/strided_slice_3/stack_1
lstm_13/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
lstm_13/strided_slice_3/stack_2Ë
lstm_13/strided_slice_3StridedSlice3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_13/strided_slice_3/stack:output:0(lstm_13/strided_slice_3/stack_1:output:0(lstm_13/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
shrink_axis_mask2
lstm_13/strided_slice_3
lstm_13/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_13/transpose_1/permÆ
lstm_13/transpose_1	Transpose3lstm_13/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_13/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
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
 *   ?2
dropout_13/dropout/Constª
dropout_13/dropout/MulMullstm_13/transpose_1:y:0!dropout_13/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dropout_13/dropout/Mul{
dropout_13/dropout/ShapeShapelstm_13/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_13/dropout/ShapeÚ
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
dtype021
/dropout_13/dropout/random_uniform/RandomUniform
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!dropout_13/dropout/GreaterEqual/yï
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2!
dropout_13/dropout/GreaterEqual¥
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dropout_13/dropout/Cast«
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dropout_13/dropout/Mul_1¯
 dense_6/Tensordot/ReadVariableOpReadVariableOp)dense_6_tensordot_readvariableop_resource*
_output_shapes
:	ç*
dtype02"
 dense_6/Tensordot/ReadVariableOpz
dense_6/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/axes
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
dense_6/Tensordot/Shape
dense_6/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/GatherV2/axisù
dense_6/Tensordot/GatherV2GatherV2 dense_6/Tensordot/Shape:output:0dense_6/Tensordot/free:output:0(dense_6/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_6/Tensordot/GatherV2
!dense_6/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_6/Tensordot/GatherV2_1/axisÿ
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
dense_6/Tensordot/Const 
dense_6/Tensordot/ProdProd#dense_6/Tensordot/GatherV2:output:0 dense_6/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod
dense_6/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_6/Tensordot/Const_1¨
dense_6/Tensordot/Prod_1Prod%dense_6/Tensordot/GatherV2_1:output:0"dense_6/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_6/Tensordot/Prod_1
dense_6/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_6/Tensordot/concat/axisØ
dense_6/Tensordot/concatConcatV2dense_6/Tensordot/free:output:0dense_6/Tensordot/axes:output:0&dense_6/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat¬
dense_6/Tensordot/stackPackdense_6/Tensordot/Prod:output:0!dense_6/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/stack¿
dense_6/Tensordot/transpose	Transposedropout_13/dropout/Mul_1:z:0!dense_6/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
dense_6/Tensordot/transpose¿
dense_6/Tensordot/ReshapeReshapedense_6/Tensordot/transpose:y:0 dense_6/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_6/Tensordot/Reshape¾
dense_6/Tensordot/MatMulMatMul"dense_6/Tensordot/Reshape:output:0(dense_6/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Tensordot/MatMul
dense_6/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_6/Tensordot/Const_2
dense_6/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_6/Tensordot/concat_1/axiså
dense_6/Tensordot/concat_1ConcatV2#dense_6/Tensordot/GatherV2:output:0"dense_6/Tensordot/Const_2:output:0(dense_6/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_6/Tensordot/concat_1°
dense_6/TensordotReshape"dense_6/Tensordot/MatMul:product:0#dense_6/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Tensordot¤
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp§
dense_6/BiasAddBiasAdddense_6/Tensordot:output:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAdd}
dense_6/SoftmaxSoftmaxdense_6/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Softmaxx
IdentityIdentitydense_6/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÆ
NoOpNoOp^dense_6/BiasAdd/ReadVariableOp!^dense_6/Tensordot/ReadVariableOp,^lstm_12/lstm_cell_12/BiasAdd/ReadVariableOp+^lstm_12/lstm_cell_12/MatMul/ReadVariableOp-^lstm_12/lstm_cell_12/MatMul_1/ReadVariableOp^lstm_12/while,^lstm_13/lstm_cell_13/BiasAdd/ReadVariableOp+^lstm_13/lstm_cell_13/MatMul/ReadVariableOp-^lstm_13/lstm_cell_13/MatMul_1/ReadVariableOp^lstm_13/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
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
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs

f
H__inference_dropout_12_layer_call_and_return_conditional_losses_26064703

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
¬]
õ
(sequential_6_lstm_13_while_body_26063161F
Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counterL
Hsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations*
&sequential_6_lstm_13_while_placeholder,
(sequential_6_lstm_13_while_placeholder_1,
(sequential_6_lstm_13_while_placeholder_2,
(sequential_6_lstm_13_while_placeholder_3E
Asequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1_0
}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0[
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:	>^
Jsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çX
Isequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	'
#sequential_6_lstm_13_while_identity)
%sequential_6_lstm_13_while_identity_1)
%sequential_6_lstm_13_while_identity_2)
%sequential_6_lstm_13_while_identity_3)
%sequential_6_lstm_13_while_identity_4)
%sequential_6_lstm_13_while_identity_5C
?sequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1
{sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensorY
Fsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:	>\
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
çV
Gsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	¢>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpí
Lsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2N
Lsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeÑ
>sequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0&sequential_6_lstm_13_while_placeholderUsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02@
>sequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOpHsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype02?
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp«
.sequential_6/lstm_13/while/lstm_cell_13/MatMulMatMulEsequential_6/lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:0Esequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_6/lstm_13/while/lstm_cell_13/MatMul
?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOpJsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype02A
?sequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp
0sequential_6/lstm_13/while/lstm_cell_13/MatMul_1MatMul(sequential_6_lstm_13_while_placeholder_2Gsequential_6/lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_6/lstm_13/while/lstm_cell_13/MatMul_1
+sequential_6/lstm_13/while/lstm_cell_13/addAddV28sequential_6/lstm_13/while/lstm_cell_13/MatMul:product:0:sequential_6/lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+sequential_6/lstm_13/while/lstm_cell_13/add
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOpIsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02@
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp
/sequential_6/lstm_13/while/lstm_cell_13/BiasAddBiasAdd/sequential_6/lstm_13/while/lstm_cell_13/add:z:0Fsequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/sequential_6/lstm_13/while/lstm_cell_13/BiasAdd´
7sequential_6/lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :29
7sequential_6/lstm_13/while/lstm_cell_13/split/split_dimã
-sequential_6/lstm_13/while/lstm_cell_13/splitSplit@sequential_6/lstm_13/while/lstm_cell_13/split/split_dim:output:08sequential_6/lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2/
-sequential_6/lstm_13/while/lstm_cell_13/splitØ
/sequential_6/lstm_13/while/lstm_cell_13/SigmoidSigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç21
/sequential_6/lstm_13/while/lstm_cell_13/SigmoidÜ
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç23
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1õ
+sequential_6/lstm_13/while/lstm_cell_13/mulMul5sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_1:y:0(sequential_6_lstm_13_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2-
+sequential_6/lstm_13/while/lstm_cell_13/mulÏ
,sequential_6/lstm_13/while/lstm_cell_13/ReluRelu6sequential_6/lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2.
,sequential_6/lstm_13/while/lstm_cell_13/Relu
-sequential_6/lstm_13/while/lstm_cell_13/mul_1Mul3sequential_6/lstm_13/while/lstm_cell_13/Sigmoid:y:0:sequential_6/lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2/
-sequential_6/lstm_13/while/lstm_cell_13/mul_1þ
-sequential_6/lstm_13/while/lstm_cell_13/add_1AddV2/sequential_6/lstm_13/while/lstm_cell_13/mul:z:01sequential_6/lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2/
-sequential_6/lstm_13/while/lstm_cell_13/add_1Ü
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid6sequential_6/lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç23
1sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2Î
.sequential_6/lstm_13/while/lstm_cell_13/Relu_1Relu1sequential_6/lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç20
.sequential_6/lstm_13/while/lstm_cell_13/Relu_1
-sequential_6/lstm_13/while/lstm_cell_13/mul_2Mul5sequential_6/lstm_13/while/lstm_cell_13/Sigmoid_2:y:0<sequential_6/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2/
-sequential_6/lstm_13/while/lstm_cell_13/mul_2É
?sequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem(sequential_6_lstm_13_while_placeholder_1&sequential_6_lstm_13_while_placeholder1sequential_6/lstm_13/while/lstm_cell_13/mul_2:z:0*
_output_shapes
: *
element_dtype02A
?sequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItem
 sequential_6/lstm_13/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2"
 sequential_6/lstm_13/while/add/y½
sequential_6/lstm_13/while/addAddV2&sequential_6_lstm_13_while_placeholder)sequential_6/lstm_13/while/add/y:output:0*
T0*
_output_shapes
: 2 
sequential_6/lstm_13/while/add
"sequential_6/lstm_13/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2$
"sequential_6/lstm_13/while/add_1/yß
 sequential_6/lstm_13/while/add_1AddV2Bsequential_6_lstm_13_while_sequential_6_lstm_13_while_loop_counter+sequential_6/lstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2"
 sequential_6/lstm_13/while/add_1¿
#sequential_6/lstm_13/while/IdentityIdentity$sequential_6/lstm_13/while/add_1:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2%
#sequential_6/lstm_13/while/Identityç
%sequential_6/lstm_13/while/Identity_1IdentityHsequential_6_lstm_13_while_sequential_6_lstm_13_while_maximum_iterations ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_1Á
%sequential_6/lstm_13/while/Identity_2Identity"sequential_6/lstm_13/while/add:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_2î
%sequential_6/lstm_13/while/Identity_3IdentityOsequential_6/lstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0 ^sequential_6/lstm_13/while/NoOp*
T0*
_output_shapes
: 2'
%sequential_6/lstm_13/while/Identity_3â
%sequential_6/lstm_13/while/Identity_4Identity1sequential_6/lstm_13/while/lstm_cell_13/mul_2:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2'
%sequential_6/lstm_13/while/Identity_4â
%sequential_6/lstm_13/while/Identity_5Identity1sequential_6/lstm_13/while/lstm_cell_13/add_1:z:0 ^sequential_6/lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2'
%sequential_6/lstm_13/while/Identity_5Ç
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
%sequential_6_lstm_13_while_identity_5.sequential_6/lstm_13/while/Identity_5:output:0"
Gsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resourceIsequential_6_lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0"
Hsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resourceJsequential_6_lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0"
Fsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resourceHsequential_6_lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"
?sequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1Asequential_6_lstm_13_while_sequential_6_lstm_13_strided_slice_1_0"ü
{sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor}sequential_6_lstm_13_while_tensorarrayv2read_tensorlistgetitem_sequential_6_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2
>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp>sequential_6/lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp2~
=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp=sequential_6/lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp2
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
Ô!
ý
E__inference_dense_6_layer_call_and_return_conditional_losses_26064901

inputs4
!tensordot_readvariableop_resource:	ç-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	ç*
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
:ÿÿÿÿÿÿÿÿÿç2
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
:ÿÿÿÿÿÿÿÿÿç: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs
ÏJ
Ò

lstm_12_while_body_26065890,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3+
'lstm_12_while_lstm_12_strided_slice_1_0g
clstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0:	]øP
=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øK
<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø
lstm_12_while_identity
lstm_12_while_identity_1
lstm_12_while_identity_2
lstm_12_while_identity_3
lstm_12_while_identity_4
lstm_12_while_identity_5)
%lstm_12_while_lstm_12_strided_slice_1e
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorL
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource:	]øN
;lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource:	>øI
:lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp¢0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp¢2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpÓ
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2A
?lstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_12/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0lstm_12_while_placeholderHlstm_12/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype023
1lstm_12/while/TensorArrayV2Read/TensorListGetItemá
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype022
0lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp÷
!lstm_12/while/lstm_cell_12/MatMulMatMul8lstm_12/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_12/while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2#
!lstm_12/while/lstm_cell_12/MatMulç
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp=lstm_12_while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype024
2lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOpà
#lstm_12/while/lstm_cell_12/MatMul_1MatMullstm_12_while_placeholder_2:lstm_12/while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2%
#lstm_12/while/lstm_cell_12/MatMul_1Ø
lstm_12/while/lstm_cell_12/addAddV2+lstm_12/while/lstm_cell_12/MatMul:product:0-lstm_12/while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2 
lstm_12/while/lstm_cell_12/addà
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp<lstm_12_while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype023
1lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOpå
"lstm_12/while/lstm_cell_12/BiasAddBiasAdd"lstm_12/while/lstm_cell_12/add:z:09lstm_12/while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2$
"lstm_12/while/lstm_cell_12/BiasAdd
*lstm_12/while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_12/while/lstm_cell_12/split/split_dim«
 lstm_12/while/lstm_cell_12/splitSplit3lstm_12/while/lstm_cell_12/split/split_dim:output:0+lstm_12/while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2"
 lstm_12/while/lstm_cell_12/split°
"lstm_12/while/lstm_cell_12/SigmoidSigmoid)lstm_12/while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2$
"lstm_12/while/lstm_cell_12/Sigmoid´
$lstm_12/while/lstm_cell_12/Sigmoid_1Sigmoid)lstm_12/while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2&
$lstm_12/while/lstm_cell_12/Sigmoid_1À
lstm_12/while/lstm_cell_12/mulMul(lstm_12/while/lstm_cell_12/Sigmoid_1:y:0lstm_12_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2 
lstm_12/while/lstm_cell_12/mul§
lstm_12/while/lstm_cell_12/ReluRelu)lstm_12/while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2!
lstm_12/while/lstm_cell_12/ReluÔ
 lstm_12/while/lstm_cell_12/mul_1Mul&lstm_12/while/lstm_cell_12/Sigmoid:y:0-lstm_12/while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2"
 lstm_12/while/lstm_cell_12/mul_1É
 lstm_12/while/lstm_cell_12/add_1AddV2"lstm_12/while/lstm_cell_12/mul:z:0$lstm_12/while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2"
 lstm_12/while/lstm_cell_12/add_1´
$lstm_12/while/lstm_cell_12/Sigmoid_2Sigmoid)lstm_12/while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2&
$lstm_12/while/lstm_cell_12/Sigmoid_2¦
!lstm_12/while/lstm_cell_12/Relu_1Relu$lstm_12/while/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2#
!lstm_12/while/lstm_cell_12/Relu_1Ø
 lstm_12/while/lstm_cell_12/mul_2Mul(lstm_12/while/lstm_cell_12/Sigmoid_2:y:0/lstm_12/while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2"
 lstm_12/while/lstm_cell_12/mul_2
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
lstm_12/while/add/y
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
lstm_12/while/add_1/y
lstm_12/while/add_1AddV2(lstm_12_while_lstm_12_while_loop_counterlstm_12/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_12/while/add_1
lstm_12/while/IdentityIdentitylstm_12/while/add_1:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity¦
lstm_12/while/Identity_1Identity.lstm_12_while_lstm_12_while_maximum_iterations^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_1
lstm_12/while/Identity_2Identitylstm_12/while/add:z:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_2º
lstm_12/while/Identity_3IdentityBlstm_12/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_12/while/NoOp*
T0*
_output_shapes
: 2
lstm_12/while/Identity_3­
lstm_12/while/Identity_4Identity$lstm_12/while/lstm_cell_12/mul_2:z:0^lstm_12/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/while/Identity_4­
lstm_12/while/Identity_5Identity$lstm_12/while/lstm_cell_12/add_1:z:0^lstm_12/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_12/while/Identity_5
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
9lstm_12_while_lstm_cell_12_matmul_readvariableop_resource;lstm_12_while_lstm_cell_12_matmul_readvariableop_resource_0"È
alstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensorclstm_12_while_tensorarrayv2read_tensorlistgetitem_lstm_12_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2f
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
\

E__inference_lstm_13_layer_call_and_return_conditional_losses_26067334

inputs>
+lstm_cell_13_matmul_readvariableop_resource:	>A
-lstm_cell_13_matmul_1_readvariableop_resource:
ç;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileD
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
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2	
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
:ÿÿÿÿÿÿÿÿÿ>2
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
valueB"ÿÿÿÿ>   27
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
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul¼
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim÷
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu_1¡
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26067250*
condR
while_cond_26067249*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç2
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
:ÿÿÿÿÿÿÿÿÿç2

IdentityÈ
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
°?
Ô
while_body_26067099
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	>I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	>G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
çA
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÐ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAdd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_1¡
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Reluµ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_1ª
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Relu_1¹
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
õ

J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26063494

inputs

states
states_11
matmul_readvariableop_resource:	]ø3
 matmul_1_readvariableop_resource:	>ø.
biasadd_readvariableop_resource:	ø
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2	
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
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_namestates
ã
Í
while_cond_26063991
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_26063991___redundant_placeholder06
2while_while_cond_26063991___redundant_placeholder16
2while_while_cond_26063991___redundant_placeholder26
2while_while_cond_26063991___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
:
¦\

E__inference_lstm_12_layer_call_and_return_conditional_losses_26066357
inputs_0>
+lstm_cell_12_matmul_readvariableop_resource:	]ø@
-lstm_cell_12_matmul_1_readvariableop_resource:	>ø;
,lstm_cell_12_biasadd_readvariableop_resource:	ø
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢whileF
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
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp­
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul»
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp©
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul_1 
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/add´
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp­
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimó
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
lstm_cell_12/split
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_1
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_1
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/add_1
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu_1 
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26066273*
condR
while_cond_26066272*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2

IdentityÈ
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
¹F

E__inference_lstm_12_layer_call_and_return_conditional_losses_26063641

inputs(
lstm_cell_12_26063559:	]ø(
lstm_cell_12_26063561:	>ø$
lstm_cell_12_26063563:	ø
identity¢$lstm_cell_12/StatefulPartitionedCall¢whileD
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
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2	
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
$lstm_cell_12/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_12_26063559lstm_cell_12_26063561lstm_cell_12_26063563*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_260634942&
$lstm_cell_12/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_12_26063559lstm_cell_12_26063561lstm_cell_12_26063563*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26063572*
condR
while_cond_26063571*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2

Identity}
NoOpNoOp%^lstm_cell_12/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2L
$lstm_cell_12/StatefulPartitionedCall$lstm_cell_12/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs


Ê
/__inference_sequential_6_layer_call_fn_26066185

inputs
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
	unknown_2:	>
	unknown_3:
ç
	unknown_4:	
	unknown_5:	ç
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_260649082
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
while_body_26064771
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_13_matmul_readvariableop_resource_0:	>I
5while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çC
4while_lstm_cell_13_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_13_matmul_readvariableop_resource:	>G
3while_lstm_cell_13_matmul_1_readvariableop_resource:
çA
2while_lstm_cell_13_biasadd_readvariableop_resource:	¢)while/lstm_cell_13/BiasAdd/ReadVariableOp¢(while/lstm_cell_13/MatMul/ReadVariableOp¢*while/lstm_cell_13/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÉ
(while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype02*
(while/lstm_cell_13/MatMul/ReadVariableOp×
while/lstm_cell_13/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMulÐ
*while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype02,
*while/lstm_cell_13/MatMul_1/ReadVariableOpÀ
while/lstm_cell_13/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/MatMul_1¸
while/lstm_cell_13/addAddV2#while/lstm_cell_13/MatMul:product:0%while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/addÈ
)while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02+
)while/lstm_cell_13/BiasAdd/ReadVariableOpÅ
while/lstm_cell_13/BiasAddBiasAddwhile/lstm_cell_13/add:z:01while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_13/BiasAdd
"while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_13/split/split_dim
while/lstm_cell_13/splitSplit+while/lstm_cell_13/split/split_dim:output:0#while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
while/lstm_cell_13/split
while/lstm_cell_13/SigmoidSigmoid!while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid
while/lstm_cell_13/Sigmoid_1Sigmoid!while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_1¡
while/lstm_cell_13/mulMul while/lstm_cell_13/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul
while/lstm_cell_13/ReluRelu!while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Reluµ
while/lstm_cell_13/mul_1Mulwhile/lstm_cell_13/Sigmoid:y:0%while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_1ª
while/lstm_cell_13/add_1AddV2while/lstm_cell_13/mul:z:0while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/add_1
while/lstm_cell_13/Sigmoid_2Sigmoid!while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Sigmoid_2
while/lstm_cell_13/Relu_1Reluwhile/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/Relu_1¹
while/lstm_cell_13/mul_2Mul while/lstm_cell_13/Sigmoid_2:y:0'while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/lstm_cell_13/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_13/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_13/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2V
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
ú	
È
&__inference_signature_wrapper_26065496
lstm_12_input
unknown:	]ø
	unknown_0:	>ø
	unknown_1:	ø
	unknown_2:	>
	unknown_3:
ç
	unknown_4:	
	unknown_5:	ç
	unknown_6:
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCalllstm_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_260632732
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
_user_specified_namelstm_12_input
¦\

E__inference_lstm_12_layer_call_and_return_conditional_losses_26066508
inputs_0>
+lstm_cell_12_matmul_readvariableop_resource:	]ø@
-lstm_cell_12_matmul_1_readvariableop_resource:	>ø;
,lstm_cell_12_biasadd_readvariableop_resource:	ø
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢whileF
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
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp­
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul»
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp©
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul_1 
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/add´
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp­
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimó
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
lstm_cell_12/split
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_1
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_1
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/add_1
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu_1 
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26066424*
condR
while_cond_26066423*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2

IdentityÈ
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0

f
H__inference_dropout_13_layer_call_and_return_conditional_losses_26064868

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿç:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
 
_user_specified_nameinputs
?
Ò
while_body_26066575
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0F
3while_lstm_cell_12_matmul_readvariableop_resource_0:	]øH
5while_lstm_cell_12_matmul_1_readvariableop_resource_0:	>øC
4while_lstm_cell_12_biasadd_readvariableop_resource_0:	ø
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorD
1while_lstm_cell_12_matmul_readvariableop_resource:	]øF
3while_lstm_cell_12_matmul_1_readvariableop_resource:	>øA
2while_lstm_cell_12_biasadd_readvariableop_resource:	ø¢)while/lstm_cell_12/BiasAdd/ReadVariableOp¢(while/lstm_cell_12/MatMul/ReadVariableOp¢*while/lstm_cell_12/MatMul_1/ReadVariableOpÃ
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
(while/lstm_cell_12/MatMul/ReadVariableOpReadVariableOp3while_lstm_cell_12_matmul_readvariableop_resource_0*
_output_shapes
:	]ø*
dtype02*
(while/lstm_cell_12/MatMul/ReadVariableOp×
while/lstm_cell_12/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:00while/lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMulÏ
*while/lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp5while_lstm_cell_12_matmul_1_readvariableop_resource_0*
_output_shapes
:	>ø*
dtype02,
*while/lstm_cell_12/MatMul_1/ReadVariableOpÀ
while/lstm_cell_12/MatMul_1MatMulwhile_placeholder_22while/lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/MatMul_1¸
while/lstm_cell_12/addAddV2#while/lstm_cell_12/MatMul:product:0%while/lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/addÈ
)while/lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp4while_lstm_cell_12_biasadd_readvariableop_resource_0*
_output_shapes	
:ø*
dtype02+
)while/lstm_cell_12/BiasAdd/ReadVariableOpÅ
while/lstm_cell_12/BiasAddBiasAddwhile/lstm_cell_12/add:z:01while/lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
while/lstm_cell_12/BiasAdd
"while/lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"while/lstm_cell_12/split/split_dim
while/lstm_cell_12/splitSplit+while/lstm_cell_12/split/split_dim:output:0#while/lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
while/lstm_cell_12/split
while/lstm_cell_12/SigmoidSigmoid!while/lstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid
while/lstm_cell_12/Sigmoid_1Sigmoid!while/lstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_1 
while/lstm_cell_12/mulMul while/lstm_cell_12/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul
while/lstm_cell_12/ReluRelu!while/lstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu´
while/lstm_cell_12/mul_1Mulwhile/lstm_cell_12/Sigmoid:y:0%while/lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_1©
while/lstm_cell_12/add_1AddV2while/lstm_cell_12/mul:z:0while/lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/add_1
while/lstm_cell_12/Sigmoid_2Sigmoid!while/lstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Sigmoid_2
while/lstm_cell_12/Relu_1Reluwhile/lstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/Relu_1¸
while/lstm_cell_12/mul_2Mul while/lstm_cell_12/Sigmoid_2:y:0'while/lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/lstm_cell_12/mul_2à
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_12/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_12/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
while/Identity_5Þ

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
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : 2V
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
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
: 
ç[

E__inference_lstm_12_layer_call_and_return_conditional_losses_26064690

inputs>
+lstm_cell_12_matmul_readvariableop_resource:	]ø@
-lstm_cell_12_matmul_1_readvariableop_resource:	>ø;
,lstm_cell_12_biasadd_readvariableop_resource:	ø
identity¢#lstm_cell_12/BiasAdd/ReadVariableOp¢"lstm_cell_12/MatMul/ReadVariableOp¢$lstm_cell_12/MatMul_1/ReadVariableOp¢whileD
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
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :>2
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
value	B :>2
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
:ÿÿÿÿÿÿÿÿÿ>2	
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
"lstm_cell_12/MatMul/ReadVariableOpReadVariableOp+lstm_cell_12_matmul_readvariableop_resource*
_output_shapes
:	]ø*
dtype02$
"lstm_cell_12/MatMul/ReadVariableOp­
lstm_cell_12/MatMulMatMulstrided_slice_2:output:0*lstm_cell_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul»
$lstm_cell_12/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_12_matmul_1_readvariableop_resource*
_output_shapes
:	>ø*
dtype02&
$lstm_cell_12/MatMul_1/ReadVariableOp©
lstm_cell_12/MatMul_1MatMulzeros:output:0,lstm_cell_12/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/MatMul_1 
lstm_cell_12/addAddV2lstm_cell_12/MatMul:product:0lstm_cell_12/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/add´
#lstm_cell_12/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_12_biasadd_readvariableop_resource*
_output_shapes	
:ø*
dtype02%
#lstm_cell_12/BiasAdd/ReadVariableOp­
lstm_cell_12/BiasAddBiasAddlstm_cell_12/add:z:0+lstm_cell_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿø2
lstm_cell_12/BiasAdd~
lstm_cell_12/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_12/split/split_dimó
lstm_cell_12/splitSplit%lstm_cell_12/split/split_dim:output:0lstm_cell_12/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>*
	num_split2
lstm_cell_12/split
lstm_cell_12/SigmoidSigmoidlstm_cell_12/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid
lstm_cell_12/Sigmoid_1Sigmoidlstm_cell_12/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_1
lstm_cell_12/mulMullstm_cell_12/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul}
lstm_cell_12/ReluRelulstm_cell_12/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu
lstm_cell_12/mul_1Mullstm_cell_12/Sigmoid:y:0lstm_cell_12/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_1
lstm_cell_12/add_1AddV2lstm_cell_12/mul:z:0lstm_cell_12/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/add_1
lstm_cell_12/Sigmoid_2Sigmoidlstm_cell_12/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Sigmoid_2|
lstm_cell_12/Relu_1Relulstm_cell_12/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/Relu_1 
lstm_cell_12/mul_2Mullstm_cell_12/Sigmoid_2:y:0!lstm_cell_12/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>2
lstm_cell_12/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_12_matmul_readvariableop_resource-lstm_cell_12_matmul_1_readvariableop_resource,lstm_cell_12_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26064606*
condR
while_cond_26064605*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>*
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
:ÿÿÿÿÿÿÿÿÿ>2
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
:ÿÿÿÿÿÿÿÿÿ>2

IdentityÈ
NoOpNoOp$^lstm_cell_12/BiasAdd/ReadVariableOp#^lstm_cell_12/MatMul/ReadVariableOp%^lstm_cell_12/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_12/BiasAdd/ReadVariableOp#lstm_cell_12/BiasAdd/ReadVariableOp2H
"lstm_cell_12/MatMul/ReadVariableOp"lstm_cell_12/MatMul/ReadVariableOp2L
$lstm_cell_12/MatMul_1/ReadVariableOp$lstm_cell_12/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ð

í
lstm_12_while_cond_26065889,
(lstm_12_while_lstm_12_while_loop_counter2
.lstm_12_while_lstm_12_while_maximum_iterations
lstm_12_while_placeholder
lstm_12_while_placeholder_1
lstm_12_while_placeholder_2
lstm_12_while_placeholder_3.
*lstm_12_while_less_lstm_12_strided_slice_1F
Blstm_12_while_lstm_12_while_cond_26065889___redundant_placeholder0F
Blstm_12_while_lstm_12_while_cond_26065889___redundant_placeholder1F
Blstm_12_while_lstm_12_while_cond_26065889___redundant_placeholder2F
Blstm_12_while_lstm_12_while_cond_26065889___redundant_placeholder3
lstm_12_while_identity

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
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿ>: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿ>:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>:

_output_shapes
: :

_output_shapes
:
À

$__inference__traced_restore_26068023
file_prefix2
assignvariableop_dense_6_kernel:	ç-
assignvariableop_1_dense_6_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: A
.assignvariableop_7_lstm_12_lstm_cell_12_kernel:	]øK
8assignvariableop_8_lstm_12_lstm_cell_12_recurrent_kernel:	>ø;
,assignvariableop_9_lstm_12_lstm_cell_12_bias:	øB
/assignvariableop_10_lstm_13_lstm_cell_13_kernel:	>M
9assignvariableop_11_lstm_13_lstm_cell_13_recurrent_kernel:
ç<
-assignvariableop_12_lstm_13_lstm_cell_13_bias:	#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: <
)assignvariableop_17_adam_dense_6_kernel_m:	ç5
'assignvariableop_18_adam_dense_6_bias_m:I
6assignvariableop_19_adam_lstm_12_lstm_cell_12_kernel_m:	]øS
@assignvariableop_20_adam_lstm_12_lstm_cell_12_recurrent_kernel_m:	>øC
4assignvariableop_21_adam_lstm_12_lstm_cell_12_bias_m:	øI
6assignvariableop_22_adam_lstm_13_lstm_cell_13_kernel_m:	>T
@assignvariableop_23_adam_lstm_13_lstm_cell_13_recurrent_kernel_m:
çC
4assignvariableop_24_adam_lstm_13_lstm_cell_13_bias_m:	<
)assignvariableop_25_adam_dense_6_kernel_v:	ç5
'assignvariableop_26_adam_dense_6_bias_v:I
6assignvariableop_27_adam_lstm_12_lstm_cell_12_kernel_v:	]øS
@assignvariableop_28_adam_lstm_12_lstm_cell_12_recurrent_kernel_v:	>øC
4assignvariableop_29_adam_lstm_12_lstm_cell_12_bias_v:	øI
6assignvariableop_30_adam_lstm_13_lstm_cell_13_kernel_v:	>T
@assignvariableop_31_adam_lstm_13_lstm_cell_13_recurrent_kernel_v:
çC
4assignvariableop_32_adam_lstm_13_lstm_cell_13_bias_v:	
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

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_6_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_6_biasIdentity_1:output:0"/device:CPU:0*
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
AssignVariableOp_7AssignVariableOp.assignvariableop_7_lstm_12_lstm_cell_12_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8½
AssignVariableOp_8AssignVariableOp8assignvariableop_8_lstm_12_lstm_cell_12_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9±
AssignVariableOp_9AssignVariableOp,assignvariableop_9_lstm_12_lstm_cell_12_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10·
AssignVariableOp_10AssignVariableOp/assignvariableop_10_lstm_13_lstm_cell_13_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Á
AssignVariableOp_11AssignVariableOp9assignvariableop_11_lstm_13_lstm_cell_13_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12µ
AssignVariableOp_12AssignVariableOp-assignvariableop_12_lstm_13_lstm_cell_13_biasIdentity_12:output:0"/device:CPU:0*
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
Identity_17±
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_6_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¯
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_6_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¾
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_lstm_12_lstm_cell_12_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20È
AssignVariableOp_20AssignVariableOp@assignvariableop_20_adam_lstm_12_lstm_cell_12_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¼
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_12_lstm_cell_12_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¾
AssignVariableOp_22AssignVariableOp6assignvariableop_22_adam_lstm_13_lstm_cell_13_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23È
AssignVariableOp_23AssignVariableOp@assignvariableop_23_adam_lstm_13_lstm_cell_13_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¼
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_lstm_13_lstm_cell_13_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_6_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¯
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_6_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¾
AssignVariableOp_27AssignVariableOp6assignvariableop_27_adam_lstm_12_lstm_cell_12_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28È
AssignVariableOp_28AssignVariableOp@assignvariableop_28_adam_lstm_12_lstm_cell_12_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29¼
AssignVariableOp_29AssignVariableOp4assignvariableop_29_adam_lstm_12_lstm_cell_12_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¾
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_lstm_13_lstm_cell_13_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31È
AssignVariableOp_31AssignVariableOp@assignvariableop_31_adam_lstm_13_lstm_cell_13_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¼
AssignVariableOp_32AssignVariableOp4assignvariableop_32_adam_lstm_13_lstm_cell_13_bias_vIdentity_32:output:0"/device:CPU:0*
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
Ç
ù
/__inference_lstm_cell_13_layer_call_fn_26067792

inputs
states_0
states_1
unknown:	>
	unknown_0:
ç
	unknown_1:	
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
<:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260641242
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2

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
A:ÿÿÿÿÿÿÿÿÿ>:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
"
_user_specified_name
states/1
åJ
Ô

lstm_13_while_body_26065711,
(lstm_13_while_lstm_13_while_loop_counter2
.lstm_13_while_lstm_13_while_maximum_iterations
lstm_13_while_placeholder
lstm_13_while_placeholder_1
lstm_13_while_placeholder_2
lstm_13_while_placeholder_3+
'lstm_13_while_lstm_13_strided_slice_1_0g
clstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0N
;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0:	>Q
=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0:
çK
<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0:	
lstm_13_while_identity
lstm_13_while_identity_1
lstm_13_while_identity_2
lstm_13_while_identity_3
lstm_13_while_identity_4
lstm_13_while_identity_5)
%lstm_13_while_lstm_13_strided_slice_1e
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorL
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource:	>O
;lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource:
çI
:lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource:	¢1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp¢0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp¢2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpÓ
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   2A
?lstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape
1lstm_13/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0lstm_13_while_placeholderHlstm_13/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype023
1lstm_13/while/TensorArrayV2Read/TensorListGetItemá
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOpReadVariableOp;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0*
_output_shapes
:	>*
dtype022
0lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp÷
!lstm_13/while/lstm_cell_13/MatMulMatMul8lstm_13/while/TensorArrayV2Read/TensorListGetItem:item:08lstm_13/while/lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_13/while/lstm_cell_13/MatMulè
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp=lstm_13_while_lstm_cell_13_matmul_1_readvariableop_resource_0* 
_output_shapes
:
ç*
dtype024
2lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOpà
#lstm_13/while/lstm_cell_13/MatMul_1MatMullstm_13_while_placeholder_2:lstm_13/while/lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#lstm_13/while/lstm_cell_13/MatMul_1Ø
lstm_13/while/lstm_cell_13/addAddV2+lstm_13/while/lstm_cell_13/MatMul:product:0-lstm_13/while/lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
lstm_13/while/lstm_cell_13/addà
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp<lstm_13_while_lstm_cell_13_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype023
1lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOpå
"lstm_13/while/lstm_cell_13/BiasAddBiasAdd"lstm_13/while/lstm_cell_13/add:z:09lstm_13/while/lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2$
"lstm_13/while/lstm_cell_13/BiasAdd
*lstm_13/while/lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2,
*lstm_13/while/lstm_cell_13/split/split_dim¯
 lstm_13/while/lstm_cell_13/splitSplit3lstm_13/while/lstm_cell_13/split/split_dim:output:0+lstm_13/while/lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2"
 lstm_13/while/lstm_cell_13/split±
"lstm_13/while/lstm_cell_13/SigmoidSigmoid)lstm_13/while/lstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2$
"lstm_13/while/lstm_cell_13/Sigmoidµ
$lstm_13/while/lstm_cell_13/Sigmoid_1Sigmoid)lstm_13/while/lstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2&
$lstm_13/while/lstm_cell_13/Sigmoid_1Á
lstm_13/while/lstm_cell_13/mulMul(lstm_13/while/lstm_cell_13/Sigmoid_1:y:0lstm_13_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2 
lstm_13/while/lstm_cell_13/mul¨
lstm_13/while/lstm_cell_13/ReluRelu)lstm_13/while/lstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2!
lstm_13/while/lstm_cell_13/ReluÕ
 lstm_13/while/lstm_cell_13/mul_1Mul&lstm_13/while/lstm_cell_13/Sigmoid:y:0-lstm_13/while/lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2"
 lstm_13/while/lstm_cell_13/mul_1Ê
 lstm_13/while/lstm_cell_13/add_1AddV2"lstm_13/while/lstm_cell_13/mul:z:0$lstm_13/while/lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2"
 lstm_13/while/lstm_cell_13/add_1µ
$lstm_13/while/lstm_cell_13/Sigmoid_2Sigmoid)lstm_13/while/lstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2&
$lstm_13/while/lstm_cell_13/Sigmoid_2§
!lstm_13/while/lstm_cell_13/Relu_1Relu$lstm_13/while/lstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2#
!lstm_13/while/lstm_cell_13/Relu_1Ù
 lstm_13/while/lstm_cell_13/mul_2Mul(lstm_13/while/lstm_cell_13/Sigmoid_2:y:0/lstm_13/while/lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2"
 lstm_13/while/lstm_cell_13/mul_2
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
lstm_13/while/add/y
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
lstm_13/while/add_1/y
lstm_13/while/add_1AddV2(lstm_13_while_lstm_13_while_loop_counterlstm_13/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_13/while/add_1
lstm_13/while/IdentityIdentitylstm_13/while/add_1:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity¦
lstm_13/while/Identity_1Identity.lstm_13_while_lstm_13_while_maximum_iterations^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_1
lstm_13/while/Identity_2Identitylstm_13/while/add:z:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_2º
lstm_13/while/Identity_3IdentityBlstm_13/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_13/while/NoOp*
T0*
_output_shapes
: 2
lstm_13/while/Identity_3®
lstm_13/while/Identity_4Identity$lstm_13/while/lstm_cell_13/mul_2:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/while/Identity_4®
lstm_13/while/Identity_5Identity$lstm_13/while/lstm_cell_13/add_1:z:0^lstm_13/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_13/while/Identity_5
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
9lstm_13_while_lstm_cell_13_matmul_readvariableop_resource;lstm_13_while_lstm_cell_13_matmul_readvariableop_resource_0"È
alstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensorclstm_13_while_tensorarrayv2read_tensorlistgetitem_lstm_13_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2f
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
\

E__inference_lstm_13_layer_call_and_return_conditional_losses_26065124

inputs>
+lstm_cell_13_matmul_readvariableop_resource:	>A
-lstm_cell_13_matmul_1_readvariableop_resource:
ç;
,lstm_cell_13_biasadd_readvariableop_resource:	
identity¢#lstm_cell_13/BiasAdd/ReadVariableOp¢"lstm_cell_13/MatMul/ReadVariableOp¢$lstm_cell_13/MatMul_1/ReadVariableOp¢whileD
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
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2	
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
:ÿÿÿÿÿÿÿÿÿ>2
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
valueB"ÿÿÿÿ>   27
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
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
strided_slice_2µ
"lstm_cell_13/MatMul/ReadVariableOpReadVariableOp+lstm_cell_13_matmul_readvariableop_resource*
_output_shapes
:	>*
dtype02$
"lstm_cell_13/MatMul/ReadVariableOp­
lstm_cell_13/MatMulMatMulstrided_slice_2:output:0*lstm_cell_13/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul¼
$lstm_cell_13/MatMul_1/ReadVariableOpReadVariableOp-lstm_cell_13_matmul_1_readvariableop_resource* 
_output_shapes
:
ç*
dtype02&
$lstm_cell_13/MatMul_1/ReadVariableOp©
lstm_cell_13/MatMul_1MatMulzeros:output:0,lstm_cell_13/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/MatMul_1 
lstm_cell_13/addAddV2lstm_cell_13/MatMul:product:0lstm_cell_13/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/add´
#lstm_cell_13/BiasAdd/ReadVariableOpReadVariableOp,lstm_cell_13_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#lstm_cell_13/BiasAdd/ReadVariableOp­
lstm_cell_13/BiasAddBiasAddlstm_cell_13/add:z:0+lstm_cell_13/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_13/BiasAdd~
lstm_cell_13/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_13/split/split_dim÷
lstm_cell_13/splitSplit%lstm_cell_13/split/split_dim:output:0lstm_cell_13/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*
	num_split2
lstm_cell_13/split
lstm_cell_13/SigmoidSigmoidlstm_cell_13/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid
lstm_cell_13/Sigmoid_1Sigmoidlstm_cell_13/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_1
lstm_cell_13/mulMullstm_cell_13/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul~
lstm_cell_13/ReluRelulstm_cell_13/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu
lstm_cell_13/mul_1Mullstm_cell_13/Sigmoid:y:0lstm_cell_13/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_1
lstm_cell_13/add_1AddV2lstm_cell_13/mul:z:0lstm_cell_13/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/add_1
lstm_cell_13/Sigmoid_2Sigmoidlstm_cell_13/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Sigmoid_2}
lstm_cell_13/Relu_1Relulstm_cell_13/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/Relu_1¡
lstm_cell_13/mul_2Mullstm_cell_13/Sigmoid_2:y:0!lstm_cell_13/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
lstm_cell_13/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0+lstm_cell_13_matmul_readvariableop_resource-lstm_cell_13_matmul_1_readvariableop_resource,lstm_cell_13_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26065040*
condR
while_cond_26065039*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç2
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
:ÿÿÿÿÿÿÿÿÿç2

IdentityÈ
NoOpNoOp$^lstm_cell_13/BiasAdd/ReadVariableOp#^lstm_cell_13/MatMul/ReadVariableOp%^lstm_cell_13/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ>: : : 2J
#lstm_cell_13/BiasAdd/ReadVariableOp#lstm_cell_13/BiasAdd/ReadVariableOp2H
"lstm_cell_13/MatMul/ReadVariableOp"lstm_cell_13/MatMul/ReadVariableOp2L
$lstm_cell_13/MatMul_1/ReadVariableOp$lstm_cell_13/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>
 
_user_specified_nameinputs
&
ó
while_body_26064202
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_00
while_lstm_cell_13_26064226_0:	>1
while_lstm_cell_13_26064228_0:
ç,
while_lstm_cell_13_26064230_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor.
while_lstm_cell_13_26064226:	>/
while_lstm_cell_13_26064228:
ç*
while_lstm_cell_13_26064230:	¢*while/lstm_cell_13/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ>   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemì
*while/lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_13_26064226_0while_lstm_cell_13_26064228_0while_lstm_cell_13_26064230_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260641242,
*while/lstm_cell_13/StatefulPartitionedCall÷
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
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 2
while/Identity_3¥
while/Identity_4Identity3while/lstm_cell_13/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_4¥
while/Identity_5Identity3while/lstm_cell_13/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç2
while/Identity_5

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
while_lstm_cell_13_26064226while_lstm_cell_13_26064226_0"<
while_lstm_cell_13_26064228while_lstm_cell_13_26064228_0"<
while_lstm_cell_13_26064230while_lstm_cell_13_26064230_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : 2X
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
:ÿÿÿÿÿÿÿÿÿç:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿç:

_output_shapes
: :

_output_shapes
: 
ËF

E__inference_lstm_13_layer_call_and_return_conditional_losses_26064061

inputs(
lstm_cell_13_26063979:	>)
lstm_cell_13_26063981:
ç$
lstm_cell_13_26063983:	
identity¢$lstm_cell_13/StatefulPartitionedCall¢whileD
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
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :ç2
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
B :ç2
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
:ÿÿÿÿÿÿÿÿÿç2	
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>2
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
valueB"ÿÿÿÿ>   27
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
:ÿÿÿÿÿÿÿÿÿ>*
shrink_axis_mask2
strided_slice_2¨
$lstm_cell_13/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_13_26063979lstm_cell_13_26063981lstm_cell_13_26063983*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_260639782&
$lstm_cell_13/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   2
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
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_13_26063979lstm_cell_13_26063981lstm_cell_13_26063983*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_26063992*
condR
while_cond_26063991*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿç:ÿÿÿÿÿÿÿÿÿç: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿç   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç*
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
:ÿÿÿÿÿÿÿÿÿç*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç2

Identity}
NoOpNoOp%^lstm_cell_13/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>: : : 2L
$lstm_cell_13/StatefulPartitionedCall$lstm_cell_13/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
 
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
lstm_12_input:
serving_default_lstm_12_input:0ÿÿÿÿÿÿÿÿÿ]?
dense_64
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ñ»
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
+&call_and_return_all_conditional_losses
__call__
_default_save_signature"
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
regularization_losses
trainable_variables
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
	variables
regularization_losses
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_rnn_layer
§
regularization_losses
trainable_variables
	variables
	keras_api
+&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
½

 kernel
!bias
"regularization_losses
#trainable_variables
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
1non_trainable_variables
2layer_regularization_losses
3metrics

4layers
trainable_variables
5layer_metrics
	variables
	regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
7regularization_losses
8trainable_variables
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
5
+0
,1
-2"
trackable_list_wrapper
 "
trackable_list_wrapper
¼
;non_trainable_variables
<layer_regularization_losses
=metrics

>layers
trainable_variables
?layer_metrics
	variables

@states
regularization_losses
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
Anon_trainable_variables
Blayer_regularization_losses
Cmetrics
regularization_losses
trainable_variables
Dlayer_metrics
	variables

Elayers
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
Gregularization_losses
Htrainable_variables
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
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
¼
Knon_trainable_variables
Llayer_regularization_losses
Mmetrics

Nlayers
trainable_variables
Olayer_metrics
	variables

Pstates
regularization_losses
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
Qnon_trainable_variables
Rlayer_regularization_losses
Smetrics
regularization_losses
trainable_variables
Tlayer_metrics
	variables

Ulayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	ç2dense_6/kernel
:2dense_6/bias
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
°
Vnon_trainable_variables
Wlayer_regularization_losses
Xmetrics
"regularization_losses
#trainable_variables
Ylayer_metrics
$	variables

Zlayers
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
.:,	]ø2lstm_12/lstm_cell_12/kernel
8:6	>ø2%lstm_12/lstm_cell_12/recurrent_kernel
(:&ø2lstm_12/lstm_cell_12/bias
.:,	>2lstm_13/lstm_cell_13/kernel
9:7
ç2%lstm_13/lstm_cell_13/recurrent_kernel
(:&2lstm_13/lstm_cell_13/bias
 "
trackable_list_wrapper
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
5
+0
,1
-2"
trackable_list_wrapper
°
]non_trainable_variables
^layer_regularization_losses
_metrics
7regularization_losses
8trainable_variables
`layer_metrics
9	variables

alayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
°
bnon_trainable_variables
clayer_regularization_losses
dmetrics
Gregularization_losses
Htrainable_variables
elayer_metrics
I	variables

flayers
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
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
&:$	ç2Adam/dense_6/kernel/m
:2Adam/dense_6/bias/m
3:1	]ø2"Adam/lstm_12/lstm_cell_12/kernel/m
=:;	>ø2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/m
-:+ø2 Adam/lstm_12/lstm_cell_12/bias/m
3:1	>2"Adam/lstm_13/lstm_cell_13/kernel/m
>:<
ç2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/m
-:+2 Adam/lstm_13/lstm_cell_13/bias/m
&:$	ç2Adam/dense_6/kernel/v
:2Adam/dense_6/bias/v
3:1	]ø2"Adam/lstm_12/lstm_cell_12/kernel/v
=:;	>ø2,Adam/lstm_12/lstm_cell_12/recurrent_kernel/v
-:+ø2 Adam/lstm_12/lstm_cell_12/bias/v
3:1	>2"Adam/lstm_13/lstm_cell_13/kernel/v
>:<
ç2,Adam/lstm_13/lstm_cell_13/recurrent_kernel/v
-:+2 Adam/lstm_13/lstm_cell_13/bias/v
ö2ó
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065823
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066164
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065442
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065467À
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
/__inference_sequential_6_layer_call_fn_26064927
/__inference_sequential_6_layer_call_fn_26066185
/__inference_sequential_6_layer_call_fn_26066206
/__inference_sequential_6_layer_call_fn_26065417À
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
#__inference__wrapped_model_26063273lstm_12_input"
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
÷2ô
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066357
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066508
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066659
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066810Õ
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
*__inference_lstm_12_layer_call_fn_26066821
*__inference_lstm_12_layer_call_fn_26066832
*__inference_lstm_12_layer_call_fn_26066843
*__inference_lstm_12_layer_call_fn_26066854Õ
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
H__inference_dropout_12_layer_call_and_return_conditional_losses_26066859
H__inference_dropout_12_layer_call_and_return_conditional_losses_26066871´
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
-__inference_dropout_12_layer_call_fn_26066876
-__inference_dropout_12_layer_call_fn_26066881´
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
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067032
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067183
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067334
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067485Õ
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
*__inference_lstm_13_layer_call_fn_26067496
*__inference_lstm_13_layer_call_fn_26067507
*__inference_lstm_13_layer_call_fn_26067518
*__inference_lstm_13_layer_call_fn_26067529Õ
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
H__inference_dropout_13_layer_call_and_return_conditional_losses_26067534
H__inference_dropout_13_layer_call_and_return_conditional_losses_26067546´
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
-__inference_dropout_13_layer_call_fn_26067551
-__inference_dropout_13_layer_call_fn_26067556´
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
E__inference_dense_6_layer_call_and_return_conditional_losses_26067587¢
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
*__inference_dense_6_layer_call_fn_26067596¢
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
&__inference_signature_wrapper_26065496lstm_12_input"
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
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26067628
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26067660¾
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
/__inference_lstm_cell_12_layer_call_fn_26067677
/__inference_lstm_cell_12_layer_call_fn_26067694¾
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
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26067726
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26067758¾
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
/__inference_lstm_cell_13_layer_call_fn_26067775
/__inference_lstm_cell_13_layer_call_fn_26067792¾
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
#__inference__wrapped_model_26063273}+,-./0 !:¢7
0¢-
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ]
ª "5ª2
0
dense_6%"
dense_6ÿÿÿÿÿÿÿÿÿ®
E__inference_dense_6_layer_call_and_return_conditional_losses_26067587e !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿç
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_6_layer_call_fn_26067596X !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿç
ª "ÿÿÿÿÿÿÿÿÿ°
H__inference_dropout_12_layer_call_and_return_conditional_losses_26066859d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ>
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ>
 °
H__inference_dropout_12_layer_call_and_return_conditional_losses_26066871d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ>
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ>
 
-__inference_dropout_12_layer_call_fn_26066876W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ>
p 
ª "ÿÿÿÿÿÿÿÿÿ>
-__inference_dropout_12_layer_call_fn_26066881W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿ>
p
ª "ÿÿÿÿÿÿÿÿÿ>²
H__inference_dropout_13_layer_call_and_return_conditional_losses_26067534f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿç
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿç
 ²
H__inference_dropout_13_layer_call_and_return_conditional_losses_26067546f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿç
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿç
 
-__inference_dropout_13_layer_call_fn_26067551Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿç
p 
ª "ÿÿÿÿÿÿÿÿÿç
-__inference_dropout_13_layer_call_fn_26067556Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿç
p
ª "ÿÿÿÿÿÿÿÿÿçÔ
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066357+,-O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
 Ô
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066508+,-O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
 º
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066659q+,-?¢<
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
0ÿÿÿÿÿÿÿÿÿ>
 º
E__inference_lstm_12_layer_call_and_return_conditional_losses_26066810q+,-?¢<
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
0ÿÿÿÿÿÿÿÿÿ>
 «
*__inference_lstm_12_layer_call_fn_26066821}+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>«
*__inference_lstm_12_layer_call_fn_26066832}+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>
*__inference_lstm_12_layer_call_fn_26066843d+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ>
*__inference_lstm_12_layer_call_fn_26066854d+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ>Õ
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067032./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
 Õ
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067183./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
 »
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067334r./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ>

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿç
 »
E__inference_lstm_13_layer_call_and_return_conditional_losses_26067485r./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ>

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿç
 ¬
*__inference_lstm_13_layer_call_fn_26067496~./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç¬
*__inference_lstm_13_layer_call_fn_26067507~./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ>

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿç
*__inference_lstm_13_layer_call_fn_26067518e./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ>

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿç
*__inference_lstm_13_layer_call_fn_26067529e./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ>

 
p

 
ª "ÿÿÿÿÿÿÿÿÿçÌ
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26067628ý+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ>
"
states/1ÿÿÿÿÿÿÿÿÿ>
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ>
EB

0/1/0ÿÿÿÿÿÿÿÿÿ>

0/1/1ÿÿÿÿÿÿÿÿÿ>
 Ì
J__inference_lstm_cell_12_layer_call_and_return_conditional_losses_26067660ý+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ>
"
states/1ÿÿÿÿÿÿÿÿÿ>
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ>
EB

0/1/0ÿÿÿÿÿÿÿÿÿ>

0/1/1ÿÿÿÿÿÿÿÿÿ>
 ¡
/__inference_lstm_cell_12_layer_call_fn_26067677í+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ>
"
states/1ÿÿÿÿÿÿÿÿÿ>
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ>
A>

1/0ÿÿÿÿÿÿÿÿÿ>

1/1ÿÿÿÿÿÿÿÿÿ>¡
/__inference_lstm_cell_12_layer_call_fn_26067694í+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ>
"
states/1ÿÿÿÿÿÿÿÿÿ>
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ>
A>

1/0ÿÿÿÿÿÿÿÿÿ>

1/1ÿÿÿÿÿÿÿÿÿ>Ñ
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26067726./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ>
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿç
# 
states/1ÿÿÿÿÿÿÿÿÿç
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿç
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿç
 
0/1/1ÿÿÿÿÿÿÿÿÿç
 Ñ
J__inference_lstm_cell_13_layer_call_and_return_conditional_losses_26067758./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ>
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿç
# 
states/1ÿÿÿÿÿÿÿÿÿç
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿç
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿç
 
0/1/1ÿÿÿÿÿÿÿÿÿç
 ¦
/__inference_lstm_cell_13_layer_call_fn_26067775ò./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ>
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿç
# 
states/1ÿÿÿÿÿÿÿÿÿç
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿç
C@

1/0ÿÿÿÿÿÿÿÿÿç

1/1ÿÿÿÿÿÿÿÿÿç¦
/__inference_lstm_cell_13_layer_call_fn_26067792ò./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿ>
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿç
# 
states/1ÿÿÿÿÿÿÿÿÿç
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿç
C@

1/0ÿÿÿÿÿÿÿÿÿç

1/1ÿÿÿÿÿÿÿÿÿçÇ
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065442y+,-./0 !B¢?
8¢5
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Ç
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065467y+,-./0 !B¢?
8¢5
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_sequential_6_layer_call_and_return_conditional_losses_26065823r+,-./0 !;¢8
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
J__inference_sequential_6_layer_call_and_return_conditional_losses_26066164r+,-./0 !;¢8
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
/__inference_sequential_6_layer_call_fn_26064927l+,-./0 !B¢?
8¢5
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_6_layer_call_fn_26065417l+,-./0 !B¢?
8¢5
+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_6_layer_call_fn_26066185e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_6_layer_call_fn_26066206e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ¹
&__inference_signature_wrapper_26065496+,-./0 !K¢H
¢ 
Aª>
<
lstm_12_input+(
lstm_12_inputÿÿÿÿÿÿÿÿÿ]"5ª2
0
dense_6%"
dense_6ÿÿÿÿÿÿÿÿÿ