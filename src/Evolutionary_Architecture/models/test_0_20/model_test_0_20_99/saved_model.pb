²'
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
"serve*2.6.02v2.6.0-rc2-32-g919f693420e8ôÖ%
y
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¢*
shared_namedense_4/kernel
r
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes
:	¢*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
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

lstm_8/lstm_cell_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]**
shared_namelstm_8/lstm_cell_8/kernel

-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/kernel*
_output_shapes
:	]*
dtype0
£
#lstm_8/lstm_cell_8/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	G*4
shared_name%#lstm_8/lstm_cell_8/recurrent_kernel

7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_8/lstm_cell_8/recurrent_kernel*
_output_shapes
:	G*
dtype0

lstm_8/lstm_cell_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_8/lstm_cell_8/bias

+lstm_8/lstm_cell_8/bias/Read/ReadVariableOpReadVariableOplstm_8/lstm_cell_8/bias*
_output_shapes	
:*
dtype0

lstm_9/lstm_cell_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	G**
shared_namelstm_9/lstm_cell_9/kernel

-lstm_9/lstm_cell_9/kernel/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_9/kernel*
_output_shapes
:	G*
dtype0
¤
#lstm_9/lstm_cell_9/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¢*4
shared_name%#lstm_9/lstm_cell_9/recurrent_kernel

7lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_9/lstm_cell_9/recurrent_kernel* 
_output_shapes
:
¢*
dtype0

lstm_9/lstm_cell_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_9/lstm_cell_9/bias

+lstm_9/lstm_cell_9/bias/Read/ReadVariableOpReadVariableOplstm_9/lstm_cell_9/bias*
_output_shapes	
:*
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
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¢*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes
:	¢*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0

 Adam/lstm_8/lstm_cell_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]*1
shared_name" Adam/lstm_8/lstm_cell_8/kernel/m

4Adam/lstm_8/lstm_cell_8/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_8/lstm_cell_8/kernel/m*
_output_shapes
:	]*
dtype0
±
*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	G*;
shared_name,*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m
ª
>Adam/lstm_8/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m*
_output_shapes
:	G*
dtype0

Adam/lstm_8/lstm_cell_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_8/lstm_cell_8/bias/m

2Adam/lstm_8/lstm_cell_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_8/bias/m*
_output_shapes	
:*
dtype0

 Adam/lstm_9/lstm_cell_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	G*1
shared_name" Adam/lstm_9/lstm_cell_9/kernel/m

4Adam/lstm_9/lstm_cell_9/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_9/lstm_cell_9/kernel/m*
_output_shapes
:	G*
dtype0
²
*Adam/lstm_9/lstm_cell_9/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¢*;
shared_name,*Adam/lstm_9/lstm_cell_9/recurrent_kernel/m
«
>Adam/lstm_9/lstm_cell_9/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_9/lstm_cell_9/recurrent_kernel/m* 
_output_shapes
:
¢*
dtype0

Adam/lstm_9/lstm_cell_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_9/lstm_cell_9/bias/m

2Adam/lstm_9/lstm_cell_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_9/lstm_cell_9/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	¢*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes
:	¢*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0

 Adam/lstm_8/lstm_cell_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	]*1
shared_name" Adam/lstm_8/lstm_cell_8/kernel/v

4Adam/lstm_8/lstm_cell_8/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_8/lstm_cell_8/kernel/v*
_output_shapes
:	]*
dtype0
±
*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	G*;
shared_name,*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v
ª
>Adam/lstm_8/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v*
_output_shapes
:	G*
dtype0

Adam/lstm_8/lstm_cell_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_8/lstm_cell_8/bias/v

2Adam/lstm_8/lstm_cell_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_8/lstm_cell_8/bias/v*
_output_shapes	
:*
dtype0

 Adam/lstm_9/lstm_cell_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	G*1
shared_name" Adam/lstm_9/lstm_cell_9/kernel/v

4Adam/lstm_9/lstm_cell_9/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_9/lstm_cell_9/kernel/v*
_output_shapes
:	G*
dtype0
²
*Adam/lstm_9/lstm_cell_9/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
¢*;
shared_name,*Adam/lstm_9/lstm_cell_9/recurrent_kernel/v
«
>Adam/lstm_9/lstm_cell_9/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_9/lstm_cell_9/recurrent_kernel/v* 
_output_shapes
:
¢*
dtype0

Adam/lstm_9/lstm_cell_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_9/lstm_cell_9/bias/v

2Adam/lstm_9/lstm_cell_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_9/lstm_cell_9/bias/v*
_output_shapes	
:*
dtype0

NoOpNoOp
Ð7
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*7
value7Bþ6 B÷6
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
1metrics
trainable_variables
regularization_losses
2layer_regularization_losses
3non_trainable_variables

4layers
		variables
5layer_metrics
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
­
Ametrics
trainable_variables
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses

Dlayers
	variables
Elayer_metrics
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
­
Qmetrics
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
regularization_losses

Tlayers
	variables
Ulayer_metrics
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1
 

 0
!1
­
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
_]
VARIABLE_VALUElstm_8/lstm_cell_8/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_8/lstm_cell_8/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_8/lstm_cell_8/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUElstm_9/lstm_cell_9/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#lstm_9/lstm_cell_9/recurrent_kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUElstm_9/lstm_cell_9/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
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
­
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
­
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
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_8/lstm_cell_8/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lstm_8/lstm_cell_8/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_9/lstm_cell_9/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_9/lstm_cell_9/recurrent_kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lstm_9/lstm_cell_9/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_8/lstm_cell_8/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lstm_8/lstm_cell_8/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE Adam/lstm_9/lstm_cell_9/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE*Adam/lstm_9/lstm_cell_9/recurrent_kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/lstm_9/lstm_cell_9/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_8_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ]

StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_8_inputlstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/biaslstm_9/lstm_cell_9/kernel#lstm_9/lstm_cell_9/recurrent_kernellstm_9/lstm_cell_9/biasdense_4/kerneldense_4/bias*
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
&__inference_signature_wrapper_19542146
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp-lstm_8/lstm_cell_8/kernel/Read/ReadVariableOp7lstm_8/lstm_cell_8/recurrent_kernel/Read/ReadVariableOp+lstm_8/lstm_cell_8/bias/Read/ReadVariableOp-lstm_9/lstm_cell_9/kernel/Read/ReadVariableOp7lstm_9/lstm_cell_9/recurrent_kernel/Read/ReadVariableOp+lstm_9/lstm_cell_9/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp4Adam/lstm_8/lstm_cell_8/kernel/m/Read/ReadVariableOp>Adam/lstm_8/lstm_cell_8/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_8/lstm_cell_8/bias/m/Read/ReadVariableOp4Adam/lstm_9/lstm_cell_9/kernel/m/Read/ReadVariableOp>Adam/lstm_9/lstm_cell_9/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_9/lstm_cell_9/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp4Adam/lstm_8/lstm_cell_8/kernel/v/Read/ReadVariableOp>Adam/lstm_8/lstm_cell_8/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_8/lstm_cell_8/bias/v/Read/ReadVariableOp4Adam/lstm_9/lstm_cell_9/kernel/v/Read/ReadVariableOp>Adam/lstm_9/lstm_cell_9/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_9/lstm_cell_9/bias/v/Read/ReadVariableOpConst*.
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
!__inference__traced_save_19544564
þ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_8/lstm_cell_8/kernel#lstm_8/lstm_cell_8/recurrent_kernellstm_8/lstm_cell_8/biaslstm_9/lstm_cell_9/kernel#lstm_9/lstm_cell_9/recurrent_kernellstm_9/lstm_cell_9/biastotalcounttotal_1count_1Adam/dense_4/kernel/mAdam/dense_4/bias/m Adam/lstm_8/lstm_cell_8/kernel/m*Adam/lstm_8/lstm_cell_8/recurrent_kernel/mAdam/lstm_8/lstm_cell_8/bias/m Adam/lstm_9/lstm_cell_9/kernel/m*Adam/lstm_9/lstm_cell_9/recurrent_kernel/mAdam/lstm_9/lstm_cell_9/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/v Adam/lstm_8/lstm_cell_8/kernel/v*Adam/lstm_8/lstm_cell_8/recurrent_kernel/vAdam/lstm_8/lstm_cell_8/bias/v Adam/lstm_9/lstm_cell_9/kernel/v*Adam/lstm_9/lstm_cell_9/recurrent_kernel/vAdam/lstm_9/lstm_cell_9/bias/v*-
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
$__inference__traced_restore_19544673¼$
ºF

D__inference_lstm_9_layer_call_and_return_conditional_losses_19540921

inputs'
lstm_cell_9_19540839:	G(
lstm_cell_9_19540841:
¢#
lstm_cell_9_19540843:	
identity¢#lstm_cell_9/StatefulPartitionedCall¢whileD
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
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2	
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2
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
valueB"ÿÿÿÿG   27
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
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
strided_slice_2¢
#lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_9_19540839lstm_cell_9_19540841lstm_cell_9_19540843*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_195407742%
#lstm_cell_9/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2
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
while/loop_counterË
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_9_19540839lstm_cell_9_19540841lstm_cell_9_19540843*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19540852*
condR
while_cond_19540851*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2

Identity|
NoOpNoOp$^lstm_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG: : : 2J
#lstm_cell_9/StatefulPartitionedCall#lstm_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs


Ê
/__inference_sequential_4_layer_call_fn_19542835

inputs
unknown:	]
	unknown_0:	G
	unknown_1:	
	unknown_2:	G
	unknown_3:
¢
	unknown_4:	
	unknown_5:	¢
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_195415582
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

e
G__inference_dropout_9_layer_call_and_return_conditional_losses_19544184

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¢:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
Ö
f
G__inference_dropout_9_layer_call_and_return_conditional_losses_19544196

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
:ÿÿÿÿÿÿÿÿÿ¢2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¢:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
§

Ð
/__inference_sequential_4_layer_call_fn_19541577
lstm_8_input
unknown:	]
	unknown_0:	G
	unknown_1:	
	unknown_2:	G
	unknown_3:
¢
	unknown_4:	
	unknown_5:	¢
	unknown_6:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_195415582
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
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
&
_user_specified_namelstm_8_input
Ú>
Ë
while_body_19541690
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	GH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	GF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
¢@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOpÔ
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMulÍ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp½
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMul_1´
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/addÅ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOpÁ
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/BiasAdd
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dim
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
while/lstm_cell_9/split
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_1
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu±
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_1¦
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/add_1
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_2
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu_1µ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
Ä>
É
while_body_19543225
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	]G
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	]E
2while_lstm_cell_8_matmul_1_readvariableop_resource:	G@
1while_lstm_cell_8_biasadd_readvariableop_resource:	¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpÔ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMulÌ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp½
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMul_1´
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/addÅ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpÁ
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu°
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_1¥
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu_1´
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
Ú>
Ë
while_body_19544051
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	GH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	GF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
¢@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOpÔ
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMulÍ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp½
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMul_1´
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/addÅ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOpÁ
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/BiasAdd
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dim
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
while/lstm_cell_9/split
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_1
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu±
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_1¦
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/add_1
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_2
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu_1µ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 


I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_19540774

inputs

states
states_11
matmul_readvariableop_resource:	G4
 matmul_1_readvariableop_resource:
¢.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	G*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

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
A:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_namestates
°
¶
)__inference_lstm_8_layer_call_fn_19543493

inputs
unknown:	]
	unknown_0:	G
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_8_layer_call_and_return_conditional_losses_195413402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

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
while_cond_19541885
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19541885___redundant_placeholder06
2while_while_cond_19541885___redundant_placeholder16
2while_while_cond_19541885___redundant_placeholder26
2while_while_cond_19541885___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:

e
G__inference_dropout_9_layer_call_and_return_conditional_losses_19541518

inputs

identity_1_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityn

Identity_1IdentityIdentity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¢:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
Ñó
ë
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542473

inputsD
1lstm_8_lstm_cell_8_matmul_readvariableop_resource:	]F
3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:	GA
2lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	D
1lstm_9_lstm_cell_9_matmul_readvariableop_resource:	GG
3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:
¢A
2lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	<
)dense_4_tensordot_readvariableop_resource:	¢5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_4/BiasAdd/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp¢)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp¢(lstm_8/lstm_cell_8/MatMul/ReadVariableOp¢*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp¢lstm_8/while¢)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp¢(lstm_9/lstm_cell_9/MatMul/ReadVariableOp¢*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp¢lstm_9/whileR
lstm_8/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_8/Shape
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stack
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slicej
lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
lstm_8/zeros/mul/y
lstm_8/zeros/mulMullstm_8/strided_slice:output:0lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/mulm
lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_8/zeros/Less/y
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessp
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2
lstm_8/zeros/packed/1
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros/packedm
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros/Const
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/zerosn
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
lstm_8/zeros_1/mul/y
lstm_8/zeros_1/mulMullstm_8/strided_slice:output:0lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/mulq
lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_8/zeros_1/Less/y
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lesst
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2
lstm_8/zeros_1/packed/1¥
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros_1/packedq
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros_1/Const
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/zeros_1
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/perm
lstm_8/transpose	Transposeinputslstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stack
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_8/TensorArrayV2/element_shapeÎ
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2Í
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensor
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stack
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2¦
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_8/strided_slice_2Ç
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02*
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpÆ
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:00lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/lstm_cell_8/MatMulÍ
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02,
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpÂ
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/zeros:output:02lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/lstm_cell_8/MatMul_1¸
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/MatMul:product:0%lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/lstm_cell_8/addÆ
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpÅ
lstm_8/lstm_cell_8/BiasAddBiasAddlstm_8/lstm_cell_8/add:z:01lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/lstm_cell_8/BiasAdd
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_8/lstm_cell_8/split/split_dim
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0#lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
lstm_8/lstm_cell_8/split
lstm_8/lstm_cell_8/SigmoidSigmoid!lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Sigmoid
lstm_8/lstm_cell_8/Sigmoid_1Sigmoid!lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Sigmoid_1¤
lstm_8/lstm_cell_8/mulMul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/mul
lstm_8/lstm_cell_8/ReluRelu!lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Relu´
lstm_8/lstm_cell_8/mul_1Mullstm_8/lstm_cell_8/Sigmoid:y:0%lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/mul_1©
lstm_8/lstm_cell_8/add_1AddV2lstm_8/lstm_cell_8/mul:z:0lstm_8/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/add_1
lstm_8/lstm_cell_8/Sigmoid_2Sigmoid!lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Sigmoid_2
lstm_8/lstm_cell_8/Relu_1Relulstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Relu_1¸
lstm_8/lstm_cell_8/mul_2Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0'lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/mul_2
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2&
$lstm_8/TensorArrayV2_1/element_shapeÔ
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2_1\
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/time
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counterõ
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_8_lstm_cell_8_matmul_readvariableop_resource3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_8_while_body_19542213*&
condR
lstm_8_while_cond_19542212*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
lstm_8/whileÃ
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStack
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_8/strided_slice_3/stack
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2Ä
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
lstm_8/strided_slice_3
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/permÁ
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtime
dropout_8/IdentityIdentitylstm_8/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
dropout_8/Identityg
lstm_9/ShapeShapedropout_8/Identity:output:0*
T0*
_output_shapes
:2
lstm_9/Shape
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice/stack
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_1
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_2
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slicek
lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
lstm_9/zeros/mul/y
lstm_9/zeros/mulMullstm_9/strided_slice:output:0lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/mulm
lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_9/zeros/Less/y
lstm_9/zeros/LessLesslstm_9/zeros/mul:z:0lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/Lessq
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¢2
lstm_9/zeros/packed/1
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros/packedm
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros/Const
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/zeroso
lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
lstm_9/zeros_1/mul/y
lstm_9/zeros_1/mulMullstm_9/strided_slice:output:0lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/mulq
lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_9/zeros_1/Less/y
lstm_9/zeros_1/LessLesslstm_9/zeros_1/mul:z:0lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/Lessu
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¢2
lstm_9/zeros_1/packed/1¥
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros_1/packedq
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros_1/Const
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/zeros_1
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose/perm¤
lstm_9/transpose	Transposedropout_8/Identity:output:0lstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_9/transposed
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:2
lstm_9/Shape_1
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_1/stack
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_1
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_2
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slice_1
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_9/TensorArrayV2/element_shapeÎ
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2Í
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2>
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_9/TensorArrayUnstack/TensorListFromTensor
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_2/stack
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_1
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_2¦
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
lstm_9/strided_slice_2Ç
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp1lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype02*
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpÆ
lstm_9/lstm_cell_9/MatMulMatMullstm_9/strided_slice_2:output:00lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/lstm_cell_9/MatMulÎ
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02,
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpÂ
lstm_9/lstm_cell_9/MatMul_1MatMullstm_9/zeros:output:02lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/lstm_cell_9/MatMul_1¸
lstm_9/lstm_cell_9/addAddV2#lstm_9/lstm_cell_9/MatMul:product:0%lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/lstm_cell_9/addÆ
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpÅ
lstm_9/lstm_cell_9/BiasAddBiasAddlstm_9/lstm_cell_9/add:z:01lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/lstm_cell_9/BiasAdd
"lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_9/lstm_cell_9/split/split_dim
lstm_9/lstm_cell_9/splitSplit+lstm_9/lstm_cell_9/split/split_dim:output:0#lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
lstm_9/lstm_cell_9/split
lstm_9/lstm_cell_9/SigmoidSigmoid!lstm_9/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Sigmoid
lstm_9/lstm_cell_9/Sigmoid_1Sigmoid!lstm_9/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Sigmoid_1¥
lstm_9/lstm_cell_9/mulMul lstm_9/lstm_cell_9/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/mul
lstm_9/lstm_cell_9/ReluRelu!lstm_9/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Reluµ
lstm_9/lstm_cell_9/mul_1Mullstm_9/lstm_cell_9/Sigmoid:y:0%lstm_9/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/mul_1ª
lstm_9/lstm_cell_9/add_1AddV2lstm_9/lstm_cell_9/mul:z:0lstm_9/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/add_1
lstm_9/lstm_cell_9/Sigmoid_2Sigmoid!lstm_9/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Sigmoid_2
lstm_9/lstm_cell_9/Relu_1Relulstm_9/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Relu_1¹
lstm_9/lstm_cell_9/mul_2Mul lstm_9/lstm_cell_9/Sigmoid_2:y:0'lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/mul_2
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2&
$lstm_9/TensorArrayV2_1/element_shapeÔ
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2_1\
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/time
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_9/while/maximum_iterationsx
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/while/loop_counterù
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_9_lstm_cell_9_matmul_readvariableop_resource3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_9_while_body_19542361*&
condR
lstm_9_while_cond_19542360*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
lstm_9/whileÃ
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   29
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
element_dtype02+
)lstm_9/TensorArrayV2Stack/TensorListStack
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_9/strided_slice_3/stack
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_9/strided_slice_3/stack_1
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_3/stack_2Å
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
shrink_axis_mask2
lstm_9/strided_slice_3
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose_1/permÂ
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/transpose_1t
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/runtime
dropout_9/IdentityIdentitylstm_9/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dropout_9/Identity¯
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	¢*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/free}
dense_4/Tensordot/ShapeShapedropout_9/Identity:output:0*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axisù
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axisÿ
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1¨
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axisØ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat¬
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack¾
dense_4/Tensordot/transpose	Transposedropout_9/Identity:output:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dense_4/Tensordot/transpose¿
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_4/Tensordot/Reshape¾
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Tensordot/MatMul
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/Const_2
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axiså
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1°
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Tensordot¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp§
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAdd}
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Softmaxx
IdentityIdentitydense_4/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*^lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_8/lstm_cell_8/MatMul/ReadVariableOp+^lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_8/while*^lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)^lstm_9/lstm_cell_9/MatMul/ReadVariableOp+^lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2V
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp(lstm_8/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_8/whilelstm_8/while2V
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2T
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp(lstm_9/lstm_cell_9/MatMul/ReadVariableOp2X
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp2
lstm_9/whilelstm_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ä>
É
while_body_19543074
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	]G
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	]E
2while_lstm_cell_8_matmul_1_readvariableop_resource:	G@
1while_lstm_cell_8_biasadd_readvariableop_resource:	¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpÔ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMulÌ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp½
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMul_1´
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/addÅ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpÁ
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu°
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_1¥
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu_1´
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
¶[

D__inference_lstm_9_layer_call_and_return_conditional_losses_19541774

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	G@
,lstm_cell_9_matmul_1_readvariableop_resource:
¢:
+lstm_cell_9_biasadd_readvariableop_resource:	
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢whileD
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
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2	
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
:ÿÿÿÿÿÿÿÿÿG2
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
valueB"ÿÿÿÿG   27
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
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpª
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul¹
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOp¦
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul_1
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/add±
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOp©
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimó
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
lstm_cell_9/split
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_1
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_1
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/add_1
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu_1
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19541690*
condR
while_cond_19541689*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2

IdentityÅ
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
¨F

D__inference_lstm_8_layer_call_and_return_conditional_losses_19540081

inputs'
lstm_cell_8_19539999:	]'
lstm_cell_8_19540001:	G#
lstm_cell_8_19540003:	
identity¢#lstm_cell_8/StatefulPartitionedCall¢whileD
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
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2	
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
strided_slice_2
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_19539999lstm_cell_8_19540001lstm_cell_8_19540003*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_195399982%
#lstm_cell_8/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2
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
while/loop_counterÇ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_19539999lstm_cell_8_19540001lstm_cell_8_19540003*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19540012*
condR
while_cond_19540011*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2

Identity|
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ø[

D__inference_lstm_8_layer_call_and_return_conditional_losses_19543007
inputs_0=
*lstm_cell_8_matmul_readvariableop_resource:	]?
,lstm_cell_8_matmul_1_readvariableop_resource:	G:
+lstm_cell_8_biasadd_readvariableop_resource:	
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢whileF
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
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2	
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
strided_slice_2²
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpª
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul¸
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp¦
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/add±
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp©
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimï
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19542923*
condR
while_cond_19542922*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2

IdentityÅ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
Ö
f
G__inference_dropout_9_layer_call_and_return_conditional_losses_19541607

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
:ÿÿÿÿÿÿÿÿÿ¢2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¹
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dropout/Mul_1j
IdentityIdentitydropout/Mul_1:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¢:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
²

Ù
lstm_8_while_cond_19542539*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1D
@lstm_8_while_lstm_8_while_cond_19542539___redundant_placeholder0D
@lstm_8_while_lstm_8_while_cond_19542539___redundant_placeholder1D
@lstm_8_while_lstm_8_while_cond_19542539___redundant_placeholder2D
@lstm_8_while_lstm_8_while_cond_19542539___redundant_placeholder3
lstm_8_while_identity

lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
lstm_8/while/Lessr
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_8/while/Identity"7
lstm_8_while_identitylstm_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
¶

Ù
lstm_9_while_cond_19542694*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1D
@lstm_9_while_lstm_9_while_cond_19542694___redundant_placeholder0D
@lstm_9_while_lstm_9_while_cond_19542694___redundant_placeholder1D
@lstm_9_while_lstm_9_while_cond_19542694___redundant_placeholder2D
@lstm_9_while_lstm_9_while_cond_19542694___redundant_placeholder3
lstm_9_while_identity

lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2
lstm_9/while/Lessr
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_9/while/Identity"7
lstm_9_while_identitylstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:
Å
ø
.__inference_lstm_cell_9_layer_call_fn_19544425

inputs
states_0
states_1
unknown:	G
	unknown_0:
¢
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_195406282
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

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
A:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"
_user_specified_name
states/1
Å
ø
.__inference_lstm_cell_9_layer_call_fn_19544442

inputs
states_0
states_1
unknown:	G
	unknown_0:
¢
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_195407742
StatefulPartitionedCall|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

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
A:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"
_user_specified_name
states/1

±
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542117
lstm_8_input"
lstm_8_19542095:	]"
lstm_8_19542097:	G
lstm_8_19542099:	"
lstm_9_19542103:	G#
lstm_9_19542105:
¢
lstm_9_19542107:	#
dense_4_19542111:	¢
dense_4_19542113:
identity¢dense_4/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall¢lstm_8/StatefulPartitionedCall¢lstm_9/StatefulPartitionedCall­
lstm_8/StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputlstm_8_19542095lstm_8_19542097lstm_8_19542099*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_8_layer_call_and_return_conditional_losses_195419702 
lstm_8/StatefulPartitionedCall
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_195418032#
!dropout_8/StatefulPartitionedCallÌ
lstm_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0lstm_9_19542103lstm_9_19542105lstm_9_19542107*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_9_layer_call_and_return_conditional_losses_195417742 
lstm_9/StatefulPartitionedCall»
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_195416072#
!dropout_9/StatefulPartitionedCall½
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_4_19542111dense_4_19542113*
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
E__inference_dense_4_layer_call_and_return_conditional_losses_195415512!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityú
NoOpNoOp ^dense_4/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
&
_user_specified_namelstm_8_input
ÙH
«

lstm_9_while_body_19542695*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	GO
;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢I
:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensorJ
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	GM
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:
¢G
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	¢/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp¢.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp¢0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpÑ
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2@
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeý
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype022
0lstm_9/while/TensorArrayV2Read/TensorListGetItemÛ
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype020
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpð
lstm_9/while/lstm_cell_9/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_9/while/lstm_cell_9/MatMulâ
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype022
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpÙ
!lstm_9/while/lstm_cell_9/MatMul_1MatMullstm_9_while_placeholder_28lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_9/while/lstm_cell_9/MatMul_1Ð
lstm_9/while/lstm_cell_9/addAddV2)lstm_9/while/lstm_cell_9/MatMul:product:0+lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/while/lstm_cell_9/addÚ
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpÝ
 lstm_9/while/lstm_cell_9/BiasAddBiasAdd lstm_9/while/lstm_cell_9/add:z:07lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_9/while/lstm_cell_9/BiasAdd
(lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_9/while/lstm_cell_9/split/split_dim§
lstm_9/while/lstm_cell_9/splitSplit1lstm_9/while/lstm_cell_9/split/split_dim:output:0)lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2 
lstm_9/while/lstm_cell_9/split«
 lstm_9/while/lstm_cell_9/SigmoidSigmoid'lstm_9/while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2"
 lstm_9/while/lstm_cell_9/Sigmoid¯
"lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid'lstm_9/while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2$
"lstm_9/while/lstm_cell_9/Sigmoid_1º
lstm_9/while/lstm_cell_9/mulMul&lstm_9/while/lstm_cell_9/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/while/lstm_cell_9/mul¢
lstm_9/while/lstm_cell_9/ReluRelu'lstm_9/while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/while/lstm_cell_9/ReluÍ
lstm_9/while/lstm_cell_9/mul_1Mul$lstm_9/while/lstm_cell_9/Sigmoid:y:0+lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2 
lstm_9/while/lstm_cell_9/mul_1Â
lstm_9/while/lstm_cell_9/add_1AddV2 lstm_9/while/lstm_cell_9/mul:z:0"lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2 
lstm_9/while/lstm_cell_9/add_1¯
"lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid'lstm_9/while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2$
"lstm_9/while/lstm_cell_9/Sigmoid_2¡
lstm_9/while/lstm_cell_9/Relu_1Relu"lstm_9/while/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2!
lstm_9/while/lstm_cell_9/Relu_1Ñ
lstm_9/while/lstm_cell_9/mul_2Mul&lstm_9/while/lstm_cell_9/Sigmoid_2:y:0-lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2 
lstm_9/while/lstm_cell_9/mul_2
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1lstm_9_while_placeholder"lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_9/while/TensorArrayV2Write/TensorListSetItemj
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add/y
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/addn
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add_1/y
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/add_1
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity¡
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_1
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_2¶
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_3©
lstm_9/while/Identity_4Identity"lstm_9/while/lstm_cell_9/mul_2:z:0^lstm_9/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/while/Identity_4©
lstm_9/while/Identity_5Identity"lstm_9/while/lstm_cell_9/add_1:z:0^lstm_9/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/while/Identity_5þ
lstm_9/while/NoOpNoOp0^lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/^lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp1^lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_9/while/NoOp"7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"v
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"x
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"t
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"Ä
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2b
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2`
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2d
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
Û
¸
)__inference_lstm_8_layer_call_fn_19543471
inputs_0
unknown:	]
	unknown_0:	G
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_8_layer_call_and_return_conditional_losses_195400812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2

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
ã
Í
while_cond_19543597
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19543597___redundant_placeholder06
2while_while_cond_19543597___redundant_placeholder16
2while_while_cond_19543597___redundant_placeholder26
2while_while_cond_19543597___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:
Ô!
ý
E__inference_dense_4_layer_call_and_return_conditional_losses_19544237

inputs4
!tensordot_readvariableop_resource:	¢-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	¢*
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
:ÿÿÿÿÿÿÿÿÿ¢2
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
:ÿÿÿÿÿÿÿÿÿ¢: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
ã
Í
while_cond_19541420
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19541420___redundant_placeholder06
2while_while_cond_19541420___redundant_placeholder16
2while_while_cond_19541420___redundant_placeholder26
2while_while_cond_19541420___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:
Ú>
Ë
while_body_19543598
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	GH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	GF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
¢@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOpÔ
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMulÍ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp½
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMul_1´
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/addÅ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOpÁ
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/BiasAdd
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dim
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
while/lstm_cell_9/split
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_1
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu±
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_1¦
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/add_1
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_2
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu_1µ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
ÃH
©

lstm_8_while_body_19542213*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	]N
;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GI
:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensorJ
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	]L
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:	GG
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	¢/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp¢.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp¢0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpÑ
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeý
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItemÛ
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype020
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpð
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_8/while/lstm_cell_8/MatMulá
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype022
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpÙ
!lstm_8/while/lstm_cell_8/MatMul_1MatMullstm_8_while_placeholder_28lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_8/while/lstm_cell_8/MatMul_1Ð
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/MatMul:product:0+lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/while/lstm_cell_8/addÚ
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpÝ
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd lstm_8/while/lstm_cell_8/add:z:07lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_8/while/lstm_cell_8/BiasAdd
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_8/while/lstm_cell_8/split/split_dim£
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:0)lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2 
lstm_8/while/lstm_cell_8/splitª
 lstm_8/while/lstm_cell_8/SigmoidSigmoid'lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2"
 lstm_8/while/lstm_cell_8/Sigmoid®
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2$
"lstm_8/while/lstm_cell_8/Sigmoid_1¹
lstm_8/while/lstm_cell_8/mulMul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/while/lstm_cell_8/mul¡
lstm_8/while/lstm_cell_8/ReluRelu'lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/while/lstm_cell_8/ReluÌ
lstm_8/while/lstm_cell_8/mul_1Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0+lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2 
lstm_8/while/lstm_cell_8/mul_1Á
lstm_8/while/lstm_cell_8/add_1AddV2 lstm_8/while/lstm_cell_8/mul:z:0"lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2 
lstm_8/while/lstm_cell_8/add_1®
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2$
"lstm_8/while/lstm_cell_8/Sigmoid_2 
lstm_8/while/lstm_cell_8/Relu_1Relu"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2!
lstm_8/while/lstm_cell_8/Relu_1Ð
lstm_8/while/lstm_cell_8/mul_2Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0-lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2 
lstm_8/while/lstm_cell_8/mul_2
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder"lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/y
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/addn
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add_1/y
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity¡
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2¶
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3¨
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_2:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/while/Identity_4¨
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_1:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/while/Identity_5þ
lstm_8/while/NoOpNoOp0^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_8/while/NoOp"7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"v
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"Ä
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2b
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 


I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_19544408

inputs
states_0
states_11
matmul_readvariableop_resource:	G4
 matmul_1_readvariableop_resource:
¢.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	G*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

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
A:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"
_user_specified_name
states/1
ß
Í
while_cond_19543224
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19543224___redundant_placeholder06
2while_while_cond_19543224___redundant_placeholder16
2while_while_cond_19543224___redundant_placeholder26
2while_while_cond_19543224___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
ì%
ì
while_body_19540852
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_9_19540876_0:	G0
while_lstm_cell_9_19540878_0:
¢+
while_lstm_cell_9_19540880_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_9_19540876:	G.
while_lstm_cell_9_19540878:
¢)
while_lstm_cell_9_19540880:	¢)while/lstm_cell_9/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemæ
)while/lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_9_19540876_0while_lstm_cell_9_19540878_0while_lstm_cell_9_19540880_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_195407742+
)while/lstm_cell_9/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_9/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/lstm_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_4¤
while/Identity_5Identity2while/lstm_cell_9/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_9_19540876while_lstm_cell_9_19540876_0":
while_lstm_cell_9_19540878while_lstm_cell_9_19540878_0":
while_lstm_cell_9_19540880while_lstm_cell_9_19540880_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2V
)while/lstm_cell_9/StatefulPartitionedCall)while/lstm_cell_9/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
Ú>
Ë
while_body_19543900
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	GH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	GF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
¢@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOpÔ
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMulÍ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp½
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMul_1´
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/addÅ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOpÁ
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/BiasAdd
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dim
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
while/lstm_cell_9/split
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_1
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu±
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_1¦
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/add_1
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_2
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu_1µ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
á%
ê
while_body_19540222
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_8_19540246_0:	]/
while_lstm_cell_8_19540248_0:	G+
while_lstm_cell_8_19540250_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_8_19540246:	]-
while_lstm_cell_8_19540248:	G)
while_lstm_cell_8_19540250:	¢)while/lstm_cell_8/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItemã
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_19540246_0while_lstm_cell_8_19540248_0while_lstm_cell_8_19540250_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_195401442+
)while/lstm_cell_8/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
while/Identity_3£
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_4£
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_8_19540246while_lstm_cell_8_19540246_0":
while_lstm_cell_8_19540248while_lstm_cell_8_19540248_0":
while_lstm_cell_8_19540250while_lstm_cell_8_19540250_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
÷	
Ç
&__inference_signature_wrapper_19542146
lstm_8_input
unknown:	]
	unknown_0:	G
	unknown_1:	
	unknown_2:	G
	unknown_3:
¢
	unknown_4:	
	unknown_5:	¢
	unknown_6:
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
#__inference__wrapped_model_195399232
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
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
&
_user_specified_namelstm_8_input


*__inference_dense_4_layer_call_fn_19544246

inputs
unknown:	¢
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
E__inference_dense_4_layer_call_and_return_conditional_losses_195415512
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
:ÿÿÿÿÿÿÿÿÿ¢: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
Ä>
É
while_body_19541256
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	]G
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	]E
2while_lstm_cell_8_matmul_1_readvariableop_resource:	G@
1while_lstm_cell_8_biasadd_readvariableop_resource:	¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpÔ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMulÌ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp½
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMul_1´
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/addÅ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpÁ
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu°
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_1¥
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu_1´
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
ã
Í
while_cond_19543899
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19543899___redundant_placeholder06
2while_while_cond_19543899___redundant_placeholder16
2while_while_cond_19543899___redundant_placeholder26
2while_while_cond_19543899___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:
Ä>
É
while_body_19541886
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	]G
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	]E
2while_lstm_cell_8_matmul_1_readvariableop_resource:	G@
1while_lstm_cell_8_biasadd_readvariableop_resource:	¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpÔ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMulÌ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp½
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMul_1´
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/addÅ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpÁ
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu°
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_1¥
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu_1´
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
Û
¸
)__inference_lstm_8_layer_call_fn_19543482
inputs_0
unknown:	]
	unknown_0:	G
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_8_layer_call_and_return_conditional_losses_195402912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2

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
Î
f
G__inference_dropout_8_layer_call_and_return_conditional_losses_19541803

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
:ÿÿÿÿÿÿÿÿÿG2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
½
Ý
'sequential_4_lstm_9_while_cond_19539810D
@sequential_4_lstm_9_while_sequential_4_lstm_9_while_loop_counterJ
Fsequential_4_lstm_9_while_sequential_4_lstm_9_while_maximum_iterations)
%sequential_4_lstm_9_while_placeholder+
'sequential_4_lstm_9_while_placeholder_1+
'sequential_4_lstm_9_while_placeholder_2+
'sequential_4_lstm_9_while_placeholder_3F
Bsequential_4_lstm_9_while_less_sequential_4_lstm_9_strided_slice_1^
Zsequential_4_lstm_9_while_sequential_4_lstm_9_while_cond_19539810___redundant_placeholder0^
Zsequential_4_lstm_9_while_sequential_4_lstm_9_while_cond_19539810___redundant_placeholder1^
Zsequential_4_lstm_9_while_sequential_4_lstm_9_while_cond_19539810___redundant_placeholder2^
Zsequential_4_lstm_9_while_sequential_4_lstm_9_while_cond_19539810___redundant_placeholder3&
"sequential_4_lstm_9_while_identity
Ô
sequential_4/lstm_9/while/LessLess%sequential_4_lstm_9_while_placeholderBsequential_4_lstm_9_while_less_sequential_4_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_4/lstm_9/while/Less
"sequential_4/lstm_9/while/IdentityIdentity"sequential_4/lstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_4/lstm_9/while/Identity"Q
"sequential_4_lstm_9_while_identity+sequential_4/lstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:

e
G__inference_dropout_8_layer_call_and_return_conditional_losses_19541353

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
õ[

D__inference_lstm_9_layer_call_and_return_conditional_losses_19543833
inputs_0=
*lstm_cell_9_matmul_readvariableop_resource:	G@
,lstm_cell_9_matmul_1_readvariableop_resource:
¢:
+lstm_cell_9_biasadd_readvariableop_resource:	
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢whileF
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
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2	
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2
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
valueB"ÿÿÿÿG   27
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
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpª
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul¹
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOp¦
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul_1
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/add±
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOp©
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimó
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
lstm_cell_9/split
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_1
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_1
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/add_1
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu_1
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19543749*
condR
while_cond_19543748*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2

IdentityÅ
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
inputs/0
ô

I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_19540144

inputs

states
states_11
matmul_readvariableop_resource:	]3
 matmul_1_readvariableop_resource:	G.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_namestates

é
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542092
lstm_8_input"
lstm_8_19542070:	]"
lstm_8_19542072:	G
lstm_8_19542074:	"
lstm_9_19542078:	G#
lstm_9_19542080:
¢
lstm_9_19542082:	#
dense_4_19542086:	¢
dense_4_19542088:
identity¢dense_4/StatefulPartitionedCall¢lstm_8/StatefulPartitionedCall¢lstm_9/StatefulPartitionedCall­
lstm_8/StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputlstm_8_19542070lstm_8_19542072lstm_8_19542074*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_8_layer_call_and_return_conditional_losses_195413402 
lstm_8/StatefulPartitionedCallþ
dropout_8/PartitionedCallPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_195413532
dropout_8/PartitionedCallÄ
lstm_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0lstm_9_19542078lstm_9_19542080lstm_9_19542082*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_9_layer_call_and_return_conditional_losses_195415052 
lstm_9/StatefulPartitionedCallÿ
dropout_9/PartitionedCallPartitionedCall'lstm_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_195415182
dropout_9/PartitionedCallµ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_4_19542086dense_4_19542088*
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
E__inference_dense_4_layer_call_and_return_conditional_losses_195415512!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity²
NoOpNoOp ^dense_4/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
&
_user_specified_namelstm_8_input
[

D__inference_lstm_8_layer_call_and_return_conditional_losses_19541340

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	]?
,lstm_cell_8_matmul_1_readvariableop_resource:	G:
+lstm_cell_8_biasadd_readvariableop_resource:	
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢whileD
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
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2	
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
strided_slice_2²
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpª
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul¸
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp¦
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/add±
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp©
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimï
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19541256*
condR
while_cond_19541255*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG2
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
:ÿÿÿÿÿÿÿÿÿG2

IdentityÅ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ß®
´	
#__inference__wrapped_model_19539923
lstm_8_inputQ
>sequential_4_lstm_8_lstm_cell_8_matmul_readvariableop_resource:	]S
@sequential_4_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:	GN
?sequential_4_lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	Q
>sequential_4_lstm_9_lstm_cell_9_matmul_readvariableop_resource:	GT
@sequential_4_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:
¢N
?sequential_4_lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	I
6sequential_4_dense_4_tensordot_readvariableop_resource:	¢B
4sequential_4_dense_4_biasadd_readvariableop_resource:
identity¢+sequential_4/dense_4/BiasAdd/ReadVariableOp¢-sequential_4/dense_4/Tensordot/ReadVariableOp¢6sequential_4/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp¢5sequential_4/lstm_8/lstm_cell_8/MatMul/ReadVariableOp¢7sequential_4/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp¢sequential_4/lstm_8/while¢6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp¢5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp¢7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp¢sequential_4/lstm_9/whiler
sequential_4/lstm_8/ShapeShapelstm_8_input*
T0*
_output_shapes
:2
sequential_4/lstm_8/Shape
'sequential_4/lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_4/lstm_8/strided_slice/stack 
)sequential_4/lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_4/lstm_8/strided_slice/stack_1 
)sequential_4/lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_4/lstm_8/strided_slice/stack_2Ú
!sequential_4/lstm_8/strided_sliceStridedSlice"sequential_4/lstm_8/Shape:output:00sequential_4/lstm_8/strided_slice/stack:output:02sequential_4/lstm_8/strided_slice/stack_1:output:02sequential_4/lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_4/lstm_8/strided_slice
sequential_4/lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2!
sequential_4/lstm_8/zeros/mul/y¼
sequential_4/lstm_8/zeros/mulMul*sequential_4/lstm_8/strided_slice:output:0(sequential_4/lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_4/lstm_8/zeros/mul
 sequential_4/lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2"
 sequential_4/lstm_8/zeros/Less/y·
sequential_4/lstm_8/zeros/LessLess!sequential_4/lstm_8/zeros/mul:z:0)sequential_4/lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_8/zeros/Less
"sequential_4/lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2$
"sequential_4/lstm_8/zeros/packed/1Ó
 sequential_4/lstm_8/zeros/packedPack*sequential_4/lstm_8/strided_slice:output:0+sequential_4/lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_4/lstm_8/zeros/packed
sequential_4/lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_4/lstm_8/zeros/ConstÅ
sequential_4/lstm_8/zerosFill)sequential_4/lstm_8/zeros/packed:output:0(sequential_4/lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
sequential_4/lstm_8/zeros
!sequential_4/lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2#
!sequential_4/lstm_8/zeros_1/mul/yÂ
sequential_4/lstm_8/zeros_1/mulMul*sequential_4/lstm_8/strided_slice:output:0*sequential_4/lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_8/zeros_1/mul
"sequential_4/lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_4/lstm_8/zeros_1/Less/y¿
 sequential_4/lstm_8/zeros_1/LessLess#sequential_4/lstm_8/zeros_1/mul:z:0+sequential_4/lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_8/zeros_1/Less
$sequential_4/lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2&
$sequential_4/lstm_8/zeros_1/packed/1Ù
"sequential_4/lstm_8/zeros_1/packedPack*sequential_4/lstm_8/strided_slice:output:0-sequential_4/lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_4/lstm_8/zeros_1/packed
!sequential_4/lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_4/lstm_8/zeros_1/ConstÍ
sequential_4/lstm_8/zeros_1Fill+sequential_4/lstm_8/zeros_1/packed:output:0*sequential_4/lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
sequential_4/lstm_8/zeros_1
"sequential_4/lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_4/lstm_8/transpose/perm¼
sequential_4/lstm_8/transpose	Transposelstm_8_input+sequential_4/lstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
sequential_4/lstm_8/transpose
sequential_4/lstm_8/Shape_1Shape!sequential_4/lstm_8/transpose:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_8/Shape_1 
)sequential_4/lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_4/lstm_8/strided_slice_1/stack¤
+sequential_4/lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_1/stack_1¤
+sequential_4/lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_1/stack_2æ
#sequential_4/lstm_8/strided_slice_1StridedSlice$sequential_4/lstm_8/Shape_1:output:02sequential_4/lstm_8/strided_slice_1/stack:output:04sequential_4/lstm_8/strided_slice_1/stack_1:output:04sequential_4/lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_4/lstm_8/strided_slice_1­
/sequential_4/lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/sequential_4/lstm_8/TensorArrayV2/element_shape
!sequential_4/lstm_8/TensorArrayV2TensorListReserve8sequential_4/lstm_8/TensorArrayV2/element_shape:output:0,sequential_4/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_4/lstm_8/TensorArrayV2ç
Isequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2K
Isequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeÈ
;sequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/lstm_8/transpose:y:0Rsequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor 
)sequential_4/lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_4/lstm_8/strided_slice_2/stack¤
+sequential_4/lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_2/stack_1¤
+sequential_4/lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_2/stack_2ô
#sequential_4/lstm_8/strided_slice_2StridedSlice!sequential_4/lstm_8/transpose:y:02sequential_4/lstm_8/strided_slice_2/stack:output:04sequential_4/lstm_8/strided_slice_2/stack_1:output:04sequential_4/lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2%
#sequential_4/lstm_8/strided_slice_2î
5sequential_4/lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp>sequential_4_lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype027
5sequential_4/lstm_8/lstm_cell_8/MatMul/ReadVariableOpú
&sequential_4/lstm_8/lstm_cell_8/MatMulMatMul,sequential_4/lstm_8/strided_slice_2:output:0=sequential_4/lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_4/lstm_8/lstm_cell_8/MatMulô
7sequential_4/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp@sequential_4_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype029
7sequential_4/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpö
(sequential_4/lstm_8/lstm_cell_8/MatMul_1MatMul"sequential_4/lstm_8/zeros:output:0?sequential_4/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_4/lstm_8/lstm_cell_8/MatMul_1ì
#sequential_4/lstm_8/lstm_cell_8/addAddV20sequential_4/lstm_8/lstm_cell_8/MatMul:product:02sequential_4/lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_4/lstm_8/lstm_cell_8/addí
6sequential_4/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp?sequential_4_lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype028
6sequential_4/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpù
'sequential_4/lstm_8/lstm_cell_8/BiasAddBiasAdd'sequential_4/lstm_8/lstm_cell_8/add:z:0>sequential_4/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_4/lstm_8/lstm_cell_8/BiasAdd¤
/sequential_4/lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_4/lstm_8/lstm_cell_8/split/split_dim¿
%sequential_4/lstm_8/lstm_cell_8/splitSplit8sequential_4/lstm_8/lstm_cell_8/split/split_dim:output:00sequential_4/lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2'
%sequential_4/lstm_8/lstm_cell_8/split¿
'sequential_4/lstm_8/lstm_cell_8/SigmoidSigmoid.sequential_4/lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2)
'sequential_4/lstm_8/lstm_cell_8/SigmoidÃ
)sequential_4/lstm_8/lstm_cell_8/Sigmoid_1Sigmoid.sequential_4/lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2+
)sequential_4/lstm_8/lstm_cell_8/Sigmoid_1Ø
#sequential_4/lstm_8/lstm_cell_8/mulMul-sequential_4/lstm_8/lstm_cell_8/Sigmoid_1:y:0$sequential_4/lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2%
#sequential_4/lstm_8/lstm_cell_8/mul¶
$sequential_4/lstm_8/lstm_cell_8/ReluRelu.sequential_4/lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2&
$sequential_4/lstm_8/lstm_cell_8/Reluè
%sequential_4/lstm_8/lstm_cell_8/mul_1Mul+sequential_4/lstm_8/lstm_cell_8/Sigmoid:y:02sequential_4/lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2'
%sequential_4/lstm_8/lstm_cell_8/mul_1Ý
%sequential_4/lstm_8/lstm_cell_8/add_1AddV2'sequential_4/lstm_8/lstm_cell_8/mul:z:0)sequential_4/lstm_8/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2'
%sequential_4/lstm_8/lstm_cell_8/add_1Ã
)sequential_4/lstm_8/lstm_cell_8/Sigmoid_2Sigmoid.sequential_4/lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2+
)sequential_4/lstm_8/lstm_cell_8/Sigmoid_2µ
&sequential_4/lstm_8/lstm_cell_8/Relu_1Relu)sequential_4/lstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2(
&sequential_4/lstm_8/lstm_cell_8/Relu_1ì
%sequential_4/lstm_8/lstm_cell_8/mul_2Mul-sequential_4/lstm_8/lstm_cell_8/Sigmoid_2:y:04sequential_4/lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2'
%sequential_4/lstm_8/lstm_cell_8/mul_2·
1sequential_4/lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   23
1sequential_4/lstm_8/TensorArrayV2_1/element_shape
#sequential_4/lstm_8/TensorArrayV2_1TensorListReserve:sequential_4/lstm_8/TensorArrayV2_1/element_shape:output:0,sequential_4/lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_4/lstm_8/TensorArrayV2_1v
sequential_4/lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_4/lstm_8/time§
,sequential_4/lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,sequential_4/lstm_8/while/maximum_iterations
&sequential_4/lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_4/lstm_8/while/loop_counter¸
sequential_4/lstm_8/whileWhile/sequential_4/lstm_8/while/loop_counter:output:05sequential_4/lstm_8/while/maximum_iterations:output:0!sequential_4/lstm_8/time:output:0,sequential_4/lstm_8/TensorArrayV2_1:handle:0"sequential_4/lstm_8/zeros:output:0$sequential_4/lstm_8/zeros_1:output:0,sequential_4/lstm_8/strided_slice_1:output:0Ksequential_4/lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_4_lstm_8_lstm_cell_8_matmul_readvariableop_resource@sequential_4_lstm_8_lstm_cell_8_matmul_1_readvariableop_resource?sequential_4_lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_4_lstm_8_while_body_19539663*3
cond+R)
'sequential_4_lstm_8_while_cond_19539662*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
sequential_4/lstm_8/whileÝ
Dsequential_4/lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2F
Dsequential_4/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape¸
6sequential_4/lstm_8/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/lstm_8/while:output:3Msequential_4/lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype028
6sequential_4/lstm_8/TensorArrayV2Stack/TensorListStack©
)sequential_4/lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/lstm_8/strided_slice_3/stack¤
+sequential_4/lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_4/lstm_8/strided_slice_3/stack_1¤
+sequential_4/lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_8/strided_slice_3/stack_2
#sequential_4/lstm_8/strided_slice_3StridedSlice?sequential_4/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/lstm_8/strided_slice_3/stack:output:04sequential_4/lstm_8/strided_slice_3/stack_1:output:04sequential_4/lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2%
#sequential_4/lstm_8/strided_slice_3¡
$sequential_4/lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_4/lstm_8/transpose_1/permõ
sequential_4/lstm_8/transpose_1	Transpose?sequential_4/lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2!
sequential_4/lstm_8/transpose_1
sequential_4/lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_4/lstm_8/runtime©
sequential_4/dropout_8/IdentityIdentity#sequential_4/lstm_8/transpose_1:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2!
sequential_4/dropout_8/Identity
sequential_4/lstm_9/ShapeShape(sequential_4/dropout_8/Identity:output:0*
T0*
_output_shapes
:2
sequential_4/lstm_9/Shape
'sequential_4/lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential_4/lstm_9/strided_slice/stack 
)sequential_4/lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_4/lstm_9/strided_slice/stack_1 
)sequential_4/lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)sequential_4/lstm_9/strided_slice/stack_2Ú
!sequential_4/lstm_9/strided_sliceStridedSlice"sequential_4/lstm_9/Shape:output:00sequential_4/lstm_9/strided_slice/stack:output:02sequential_4/lstm_9/strided_slice/stack_1:output:02sequential_4/lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!sequential_4/lstm_9/strided_slice
sequential_4/lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2!
sequential_4/lstm_9/zeros/mul/y¼
sequential_4/lstm_9/zeros/mulMul*sequential_4/lstm_9/strided_slice:output:0(sequential_4/lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential_4/lstm_9/zeros/mul
 sequential_4/lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2"
 sequential_4/lstm_9/zeros/Less/y·
sequential_4/lstm_9/zeros/LessLess!sequential_4/lstm_9/zeros/mul:z:0)sequential_4/lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2 
sequential_4/lstm_9/zeros/Less
"sequential_4/lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¢2$
"sequential_4/lstm_9/zeros/packed/1Ó
 sequential_4/lstm_9/zeros/packedPack*sequential_4/lstm_9/strided_slice:output:0+sequential_4/lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2"
 sequential_4/lstm_9/zeros/packed
sequential_4/lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential_4/lstm_9/zeros/ConstÆ
sequential_4/lstm_9/zerosFill)sequential_4/lstm_9/zeros/packed:output:0(sequential_4/lstm_9/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
sequential_4/lstm_9/zeros
!sequential_4/lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2#
!sequential_4/lstm_9/zeros_1/mul/yÂ
sequential_4/lstm_9/zeros_1/mulMul*sequential_4/lstm_9/strided_slice:output:0*sequential_4/lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_9/zeros_1/mul
"sequential_4/lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2$
"sequential_4/lstm_9/zeros_1/Less/y¿
 sequential_4/lstm_9/zeros_1/LessLess#sequential_4/lstm_9/zeros_1/mul:z:0+sequential_4/lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2"
 sequential_4/lstm_9/zeros_1/Less
$sequential_4/lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¢2&
$sequential_4/lstm_9/zeros_1/packed/1Ù
"sequential_4/lstm_9/zeros_1/packedPack*sequential_4/lstm_9/strided_slice:output:0-sequential_4/lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2$
"sequential_4/lstm_9/zeros_1/packed
!sequential_4/lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_4/lstm_9/zeros_1/ConstÎ
sequential_4/lstm_9/zeros_1Fill+sequential_4/lstm_9/zeros_1/packed:output:0*sequential_4/lstm_9/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
sequential_4/lstm_9/zeros_1
"sequential_4/lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"sequential_4/lstm_9/transpose/permØ
sequential_4/lstm_9/transpose	Transpose(sequential_4/dropout_8/Identity:output:0+sequential_4/lstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
sequential_4/lstm_9/transpose
sequential_4/lstm_9/Shape_1Shape!sequential_4/lstm_9/transpose:y:0*
T0*
_output_shapes
:2
sequential_4/lstm_9/Shape_1 
)sequential_4/lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_4/lstm_9/strided_slice_1/stack¤
+sequential_4/lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_9/strided_slice_1/stack_1¤
+sequential_4/lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_9/strided_slice_1/stack_2æ
#sequential_4/lstm_9/strided_slice_1StridedSlice$sequential_4/lstm_9/Shape_1:output:02sequential_4/lstm_9/strided_slice_1/stack:output:04sequential_4/lstm_9/strided_slice_1/stack_1:output:04sequential_4/lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#sequential_4/lstm_9/strided_slice_1­
/sequential_4/lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ21
/sequential_4/lstm_9/TensorArrayV2/element_shape
!sequential_4/lstm_9/TensorArrayV2TensorListReserve8sequential_4/lstm_9/TensorArrayV2/element_shape:output:0,sequential_4/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02#
!sequential_4/lstm_9/TensorArrayV2ç
Isequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2K
Isequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeÈ
;sequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/lstm_9/transpose:y:0Rsequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02=
;sequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensor 
)sequential_4/lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)sequential_4/lstm_9/strided_slice_2/stack¤
+sequential_4/lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_9/strided_slice_2/stack_1¤
+sequential_4/lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_9/strided_slice_2/stack_2ô
#sequential_4/lstm_9/strided_slice_2StridedSlice!sequential_4/lstm_9/transpose:y:02sequential_4/lstm_9/strided_slice_2/stack:output:04sequential_4/lstm_9/strided_slice_2/stack_1:output:04sequential_4/lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2%
#sequential_4/lstm_9/strided_slice_2î
5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp>sequential_4_lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype027
5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOpú
&sequential_4/lstm_9/lstm_cell_9/MatMulMatMul,sequential_4/lstm_9/strided_slice_2:output:0=sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&sequential_4/lstm_9/lstm_cell_9/MatMulõ
7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp@sequential_4_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype029
7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpö
(sequential_4/lstm_9/lstm_cell_9/MatMul_1MatMul"sequential_4/lstm_9/zeros:output:0?sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_4/lstm_9/lstm_cell_9/MatMul_1ì
#sequential_4/lstm_9/lstm_cell_9/addAddV20sequential_4/lstm_9/lstm_cell_9/MatMul:product:02sequential_4/lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential_4/lstm_9/lstm_cell_9/addí
6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp?sequential_4_lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype028
6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpù
'sequential_4/lstm_9/lstm_cell_9/BiasAddBiasAdd'sequential_4/lstm_9/lstm_cell_9/add:z:0>sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'sequential_4/lstm_9/lstm_cell_9/BiasAdd¤
/sequential_4/lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential_4/lstm_9/lstm_cell_9/split/split_dimÃ
%sequential_4/lstm_9/lstm_cell_9/splitSplit8sequential_4/lstm_9/lstm_cell_9/split/split_dim:output:00sequential_4/lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2'
%sequential_4/lstm_9/lstm_cell_9/splitÀ
'sequential_4/lstm_9/lstm_cell_9/SigmoidSigmoid.sequential_4/lstm_9/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2)
'sequential_4/lstm_9/lstm_cell_9/SigmoidÄ
)sequential_4/lstm_9/lstm_cell_9/Sigmoid_1Sigmoid.sequential_4/lstm_9/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2+
)sequential_4/lstm_9/lstm_cell_9/Sigmoid_1Ù
#sequential_4/lstm_9/lstm_cell_9/mulMul-sequential_4/lstm_9/lstm_cell_9/Sigmoid_1:y:0$sequential_4/lstm_9/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2%
#sequential_4/lstm_9/lstm_cell_9/mul·
$sequential_4/lstm_9/lstm_cell_9/ReluRelu.sequential_4/lstm_9/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2&
$sequential_4/lstm_9/lstm_cell_9/Relué
%sequential_4/lstm_9/lstm_cell_9/mul_1Mul+sequential_4/lstm_9/lstm_cell_9/Sigmoid:y:02sequential_4/lstm_9/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2'
%sequential_4/lstm_9/lstm_cell_9/mul_1Þ
%sequential_4/lstm_9/lstm_cell_9/add_1AddV2'sequential_4/lstm_9/lstm_cell_9/mul:z:0)sequential_4/lstm_9/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2'
%sequential_4/lstm_9/lstm_cell_9/add_1Ä
)sequential_4/lstm_9/lstm_cell_9/Sigmoid_2Sigmoid.sequential_4/lstm_9/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2+
)sequential_4/lstm_9/lstm_cell_9/Sigmoid_2¶
&sequential_4/lstm_9/lstm_cell_9/Relu_1Relu)sequential_4/lstm_9/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2(
&sequential_4/lstm_9/lstm_cell_9/Relu_1í
%sequential_4/lstm_9/lstm_cell_9/mul_2Mul-sequential_4/lstm_9/lstm_cell_9/Sigmoid_2:y:04sequential_4/lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2'
%sequential_4/lstm_9/lstm_cell_9/mul_2·
1sequential_4/lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   23
1sequential_4/lstm_9/TensorArrayV2_1/element_shape
#sequential_4/lstm_9/TensorArrayV2_1TensorListReserve:sequential_4/lstm_9/TensorArrayV2_1/element_shape:output:0,sequential_4/lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#sequential_4/lstm_9/TensorArrayV2_1v
sequential_4/lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential_4/lstm_9/time§
,sequential_4/lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2.
,sequential_4/lstm_9/while/maximum_iterations
&sequential_4/lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential_4/lstm_9/while/loop_counter¼
sequential_4/lstm_9/whileWhile/sequential_4/lstm_9/while/loop_counter:output:05sequential_4/lstm_9/while/maximum_iterations:output:0!sequential_4/lstm_9/time:output:0,sequential_4/lstm_9/TensorArrayV2_1:handle:0"sequential_4/lstm_9/zeros:output:0$sequential_4/lstm_9/zeros_1:output:0,sequential_4/lstm_9/strided_slice_1:output:0Ksequential_4/lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_4_lstm_9_lstm_cell_9_matmul_readvariableop_resource@sequential_4_lstm_9_lstm_cell_9_matmul_1_readvariableop_resource?sequential_4_lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_4_lstm_9_while_body_19539811*3
cond+R)
'sequential_4_lstm_9_while_cond_19539810*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
sequential_4/lstm_9/whileÝ
Dsequential_4/lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2F
Dsequential_4/lstm_9/TensorArrayV2Stack/TensorListStack/element_shape¹
6sequential_4/lstm_9/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/lstm_9/while:output:3Msequential_4/lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
element_dtype028
6sequential_4/lstm_9/TensorArrayV2Stack/TensorListStack©
)sequential_4/lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/lstm_9/strided_slice_3/stack¤
+sequential_4/lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+sequential_4/lstm_9/strided_slice_3/stack_1¤
+sequential_4/lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+sequential_4/lstm_9/strided_slice_3/stack_2
#sequential_4/lstm_9/strided_slice_3StridedSlice?sequential_4/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/lstm_9/strided_slice_3/stack:output:04sequential_4/lstm_9/strided_slice_3/stack_1:output:04sequential_4/lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
shrink_axis_mask2%
#sequential_4/lstm_9/strided_slice_3¡
$sequential_4/lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$sequential_4/lstm_9/transpose_1/permö
sequential_4/lstm_9/transpose_1	Transpose?sequential_4/lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/lstm_9/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2!
sequential_4/lstm_9/transpose_1
sequential_4/lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential_4/lstm_9/runtimeª
sequential_4/dropout_9/IdentityIdentity#sequential_4/lstm_9/transpose_1:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2!
sequential_4/dropout_9/IdentityÖ
-sequential_4/dense_4/Tensordot/ReadVariableOpReadVariableOp6sequential_4_dense_4_tensordot_readvariableop_resource*
_output_shapes
:	¢*
dtype02/
-sequential_4/dense_4/Tensordot/ReadVariableOp
#sequential_4/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2%
#sequential_4/dense_4/Tensordot/axes
#sequential_4/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2%
#sequential_4/dense_4/Tensordot/free¤
$sequential_4/dense_4/Tensordot/ShapeShape(sequential_4/dropout_9/Identity:output:0*
T0*
_output_shapes
:2&
$sequential_4/dense_4/Tensordot/Shape
,sequential_4/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_4/dense_4/Tensordot/GatherV2/axisº
'sequential_4/dense_4/Tensordot/GatherV2GatherV2-sequential_4/dense_4/Tensordot/Shape:output:0,sequential_4/dense_4/Tensordot/free:output:05sequential_4/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential_4/dense_4/Tensordot/GatherV2¢
.sequential_4/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 20
.sequential_4/dense_4/Tensordot/GatherV2_1/axisÀ
)sequential_4/dense_4/Tensordot/GatherV2_1GatherV2-sequential_4/dense_4/Tensordot/Shape:output:0,sequential_4/dense_4/Tensordot/axes:output:07sequential_4/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2+
)sequential_4/dense_4/Tensordot/GatherV2_1
$sequential_4/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential_4/dense_4/Tensordot/ConstÔ
#sequential_4/dense_4/Tensordot/ProdProd0sequential_4/dense_4/Tensordot/GatherV2:output:0-sequential_4/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2%
#sequential_4/dense_4/Tensordot/Prod
&sequential_4/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential_4/dense_4/Tensordot/Const_1Ü
%sequential_4/dense_4/Tensordot/Prod_1Prod2sequential_4/dense_4/Tensordot/GatherV2_1:output:0/sequential_4/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2'
%sequential_4/dense_4/Tensordot/Prod_1
*sequential_4/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential_4/dense_4/Tensordot/concat/axis
%sequential_4/dense_4/Tensordot/concatConcatV2,sequential_4/dense_4/Tensordot/free:output:0,sequential_4/dense_4/Tensordot/axes:output:03sequential_4/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential_4/dense_4/Tensordot/concatà
$sequential_4/dense_4/Tensordot/stackPack,sequential_4/dense_4/Tensordot/Prod:output:0.sequential_4/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2&
$sequential_4/dense_4/Tensordot/stackò
(sequential_4/dense_4/Tensordot/transpose	Transpose(sequential_4/dropout_9/Identity:output:0.sequential_4/dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2*
(sequential_4/dense_4/Tensordot/transposeó
&sequential_4/dense_4/Tensordot/ReshapeReshape,sequential_4/dense_4/Tensordot/transpose:y:0-sequential_4/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential_4/dense_4/Tensordot/Reshapeò
%sequential_4/dense_4/Tensordot/MatMulMatMul/sequential_4/dense_4/Tensordot/Reshape:output:05sequential_4/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%sequential_4/dense_4/Tensordot/MatMul
&sequential_4/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential_4/dense_4/Tensordot/Const_2
,sequential_4/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential_4/dense_4/Tensordot/concat_1/axis¦
'sequential_4/dense_4/Tensordot/concat_1ConcatV20sequential_4/dense_4/Tensordot/GatherV2:output:0/sequential_4/dense_4/Tensordot/Const_2:output:05sequential_4/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2)
'sequential_4/dense_4/Tensordot/concat_1ä
sequential_4/dense_4/TensordotReshape/sequential_4/dense_4/Tensordot/MatMul:product:00sequential_4/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_4/dense_4/TensordotË
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_4/dense_4/BiasAdd/ReadVariableOpÛ
sequential_4/dense_4/BiasAddBiasAdd'sequential_4/dense_4/Tensordot:output:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_4/dense_4/BiasAdd¤
sequential_4/dense_4/SoftmaxSoftmax%sequential_4/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_4/dense_4/Softmax
IdentityIdentity&sequential_4/dense_4/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityº
NoOpNoOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp.^sequential_4/dense_4/Tensordot/ReadVariableOp7^sequential_4/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp6^sequential_4/lstm_8/lstm_cell_8/MatMul/ReadVariableOp8^sequential_4/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^sequential_4/lstm_8/while7^sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp6^sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp8^sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^sequential_4/lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2^
-sequential_4/dense_4/Tensordot/ReadVariableOp-sequential_4/dense_4/Tensordot/ReadVariableOp2p
6sequential_4/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp6sequential_4/lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2n
5sequential_4/lstm_8/lstm_cell_8/MatMul/ReadVariableOp5sequential_4/lstm_8/lstm_cell_8/MatMul/ReadVariableOp2r
7sequential_4/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp7sequential_4/lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp26
sequential_4/lstm_8/whilesequential_4/lstm_8/while2p
6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp6sequential_4/lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2n
5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp5sequential_4/lstm_9/lstm_cell_9/MatMul/ReadVariableOp2r
7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp7sequential_4/lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp26
sequential_4/lstm_9/whilesequential_4/lstm_9/while:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
&
_user_specified_namelstm_8_input
¹
Ý
'sequential_4_lstm_8_while_cond_19539662D
@sequential_4_lstm_8_while_sequential_4_lstm_8_while_loop_counterJ
Fsequential_4_lstm_8_while_sequential_4_lstm_8_while_maximum_iterations)
%sequential_4_lstm_8_while_placeholder+
'sequential_4_lstm_8_while_placeholder_1+
'sequential_4_lstm_8_while_placeholder_2+
'sequential_4_lstm_8_while_placeholder_3F
Bsequential_4_lstm_8_while_less_sequential_4_lstm_8_strided_slice_1^
Zsequential_4_lstm_8_while_sequential_4_lstm_8_while_cond_19539662___redundant_placeholder0^
Zsequential_4_lstm_8_while_sequential_4_lstm_8_while_cond_19539662___redundant_placeholder1^
Zsequential_4_lstm_8_while_sequential_4_lstm_8_while_cond_19539662___redundant_placeholder2^
Zsequential_4_lstm_8_while_sequential_4_lstm_8_while_cond_19539662___redundant_placeholder3&
"sequential_4_lstm_8_while_identity
Ô
sequential_4/lstm_8/while/LessLess%sequential_4_lstm_8_while_placeholderBsequential_4_lstm_8_while_less_sequential_4_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2 
sequential_4/lstm_8/while/Less
"sequential_4/lstm_8/while/IdentityIdentity"sequential_4/lstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2$
"sequential_4/lstm_8/while/Identity"Q
"sequential_4_lstm_8_while_identity+sequential_4/lstm_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
õ[

D__inference_lstm_9_layer_call_and_return_conditional_losses_19543682
inputs_0=
*lstm_cell_9_matmul_readvariableop_resource:	G@
,lstm_cell_9_matmul_1_readvariableop_resource:
¢:
+lstm_cell_9_biasadd_readvariableop_resource:	
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢whileF
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
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2	
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2
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
valueB"ÿÿÿÿG   27
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
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpª
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul¹
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOp¦
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul_1
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/add±
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOp©
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimó
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
lstm_cell_9/split
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_1
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_1
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/add_1
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu_1
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19543598*
condR
while_cond_19543597*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2

IdentityÅ
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
inputs/0
ó
ã
J__inference_sequential_4_layer_call_and_return_conditional_losses_19541558

inputs"
lstm_8_19541341:	]"
lstm_8_19541343:	G
lstm_8_19541345:	"
lstm_9_19541506:	G#
lstm_9_19541508:
¢
lstm_9_19541510:	#
dense_4_19541552:	¢
dense_4_19541554:
identity¢dense_4/StatefulPartitionedCall¢lstm_8/StatefulPartitionedCall¢lstm_9/StatefulPartitionedCall§
lstm_8/StatefulPartitionedCallStatefulPartitionedCallinputslstm_8_19541341lstm_8_19541343lstm_8_19541345*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_8_layer_call_and_return_conditional_losses_195413402 
lstm_8/StatefulPartitionedCallþ
dropout_8/PartitionedCallPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_195413532
dropout_8/PartitionedCallÄ
lstm_9/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0lstm_9_19541506lstm_9_19541508lstm_9_19541510*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_9_layer_call_and_return_conditional_losses_195415052 
lstm_9/StatefulPartitionedCallÿ
dropout_9/PartitionedCallPartitionedCall'lstm_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_195415182
dropout_9/PartitionedCallµ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0dense_4_19541552dense_4_19541554*
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
E__inference_dense_4_layer_call_and_return_conditional_losses_195415512!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity²
NoOpNoOp ^dense_4/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
¸
÷
.__inference_lstm_cell_8_layer_call_fn_19544327

inputs
states_0
states_1
unknown:	]
	unknown_0:	G
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_195399982
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
states/1
[
É
'sequential_4_lstm_8_while_body_19539663D
@sequential_4_lstm_8_while_sequential_4_lstm_8_while_loop_counterJ
Fsequential_4_lstm_8_while_sequential_4_lstm_8_while_maximum_iterations)
%sequential_4_lstm_8_while_placeholder+
'sequential_4_lstm_8_while_placeholder_1+
'sequential_4_lstm_8_while_placeholder_2+
'sequential_4_lstm_8_while_placeholder_3C
?sequential_4_lstm_8_while_sequential_4_lstm_8_strided_slice_1_0
{sequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_4_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	][
Hsequential_4_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GV
Gsequential_4_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	&
"sequential_4_lstm_8_while_identity(
$sequential_4_lstm_8_while_identity_1(
$sequential_4_lstm_8_while_identity_2(
$sequential_4_lstm_8_while_identity_3(
$sequential_4_lstm_8_while_identity_4(
$sequential_4_lstm_8_while_identity_5A
=sequential_4_lstm_8_while_sequential_4_lstm_8_strided_slice_1}
ysequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensorW
Dsequential_4_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	]Y
Fsequential_4_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:	GT
Esequential_4_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	¢<sequential_4/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp¢;sequential_4/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp¢=sequential_4/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpë
Ksequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2M
Ksequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeË
=sequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_lstm_8_while_placeholderTsequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype02?
=sequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem
;sequential_4/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOpFsequential_4_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02=
;sequential_4/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp¤
,sequential_4/lstm_8/while/lstm_cell_8/MatMulMatMulDsequential_4/lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_4/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_4/lstm_8/while/lstm_cell_8/MatMul
=sequential_4/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOpHsequential_4_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype02?
=sequential_4/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_1MatMul'sequential_4_lstm_8_while_placeholder_2Esequential_4/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_4/lstm_8/while/lstm_cell_8/MatMul_1
)sequential_4/lstm_8/while/lstm_cell_8/addAddV26sequential_4/lstm_8/while/lstm_cell_8/MatMul:product:08sequential_4/lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/lstm_8/while/lstm_cell_8/add
<sequential_4/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOpGsequential_4_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02>
<sequential_4/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp
-sequential_4/lstm_8/while/lstm_cell_8/BiasAddBiasAdd-sequential_4/lstm_8/while/lstm_cell_8/add:z:0Dsequential_4/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_4/lstm_8/while/lstm_cell_8/BiasAdd°
5sequential_4/lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_4/lstm_8/while/lstm_cell_8/split/split_dim×
+sequential_4/lstm_8/while/lstm_cell_8/splitSplit>sequential_4/lstm_8/while/lstm_cell_8/split/split_dim:output:06sequential_4/lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2-
+sequential_4/lstm_8/while/lstm_cell_8/splitÑ
-sequential_4/lstm_8/while/lstm_cell_8/SigmoidSigmoid4sequential_4/lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2/
-sequential_4/lstm_8/while/lstm_cell_8/SigmoidÕ
/sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid4sequential_4/lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG21
/sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_1í
)sequential_4/lstm_8/while/lstm_cell_8/mulMul3sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_1:y:0'sequential_4_lstm_8_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2+
)sequential_4/lstm_8/while/lstm_cell_8/mulÈ
*sequential_4/lstm_8/while/lstm_cell_8/ReluRelu4sequential_4/lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2,
*sequential_4/lstm_8/while/lstm_cell_8/Relu
+sequential_4/lstm_8/while/lstm_cell_8/mul_1Mul1sequential_4/lstm_8/while/lstm_cell_8/Sigmoid:y:08sequential_4/lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2-
+sequential_4/lstm_8/while/lstm_cell_8/mul_1õ
+sequential_4/lstm_8/while/lstm_cell_8/add_1AddV2-sequential_4/lstm_8/while/lstm_cell_8/mul:z:0/sequential_4/lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2-
+sequential_4/lstm_8/while/lstm_cell_8/add_1Õ
/sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid4sequential_4/lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG21
/sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_2Ç
,sequential_4/lstm_8/while/lstm_cell_8/Relu_1Relu/sequential_4/lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2.
,sequential_4/lstm_8/while/lstm_cell_8/Relu_1
+sequential_4/lstm_8/while/lstm_cell_8/mul_2Mul3sequential_4/lstm_8/while/lstm_cell_8/Sigmoid_2:y:0:sequential_4/lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2-
+sequential_4/lstm_8/while/lstm_cell_8/mul_2Ã
>sequential_4/lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_lstm_8_while_placeholder_1%sequential_4_lstm_8_while_placeholder/sequential_4/lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_4/lstm_8/while/TensorArrayV2Write/TensorListSetItem
sequential_4/lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_4/lstm_8/while/add/y¹
sequential_4/lstm_8/while/addAddV2%sequential_4_lstm_8_while_placeholder(sequential_4/lstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_4/lstm_8/while/add
!sequential_4/lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_4/lstm_8/while/add_1/yÚ
sequential_4/lstm_8/while/add_1AddV2@sequential_4_lstm_8_while_sequential_4_lstm_8_while_loop_counter*sequential_4/lstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_8/while/add_1»
"sequential_4/lstm_8/while/IdentityIdentity#sequential_4/lstm_8/while/add_1:z:0^sequential_4/lstm_8/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_4/lstm_8/while/Identityâ
$sequential_4/lstm_8/while/Identity_1IdentityFsequential_4_lstm_8_while_sequential_4_lstm_8_while_maximum_iterations^sequential_4/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_8/while/Identity_1½
$sequential_4/lstm_8/while/Identity_2Identity!sequential_4/lstm_8/while/add:z:0^sequential_4/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_8/while/Identity_2ê
$sequential_4/lstm_8/while/Identity_3IdentityNsequential_4/lstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/lstm_8/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_8/while/Identity_3Ü
$sequential_4/lstm_8/while/Identity_4Identity/sequential_4/lstm_8/while/lstm_cell_8/mul_2:z:0^sequential_4/lstm_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2&
$sequential_4/lstm_8/while/Identity_4Ü
$sequential_4/lstm_8/while/Identity_5Identity/sequential_4/lstm_8/while/lstm_cell_8/add_1:z:0^sequential_4/lstm_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2&
$sequential_4/lstm_8/while/Identity_5¿
sequential_4/lstm_8/while/NoOpNoOp=^sequential_4/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp<^sequential_4/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp>^sequential_4/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_4/lstm_8/while/NoOp"Q
"sequential_4_lstm_8_while_identity+sequential_4/lstm_8/while/Identity:output:0"U
$sequential_4_lstm_8_while_identity_1-sequential_4/lstm_8/while/Identity_1:output:0"U
$sequential_4_lstm_8_while_identity_2-sequential_4/lstm_8/while/Identity_2:output:0"U
$sequential_4_lstm_8_while_identity_3-sequential_4/lstm_8/while/Identity_3:output:0"U
$sequential_4_lstm_8_while_identity_4-sequential_4/lstm_8/while/Identity_4:output:0"U
$sequential_4_lstm_8_while_identity_5-sequential_4/lstm_8/while/Identity_5:output:0"
Esequential_4_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resourceGsequential_4_lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"
Fsequential_4_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resourceHsequential_4_lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"
Dsequential_4_lstm_8_while_lstm_cell_8_matmul_readvariableop_resourceFsequential_4_lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"
=sequential_4_lstm_8_while_sequential_4_lstm_8_strided_slice_1?sequential_4_lstm_8_while_sequential_4_lstm_8_strided_slice_1_0"ø
ysequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensor{sequential_4_lstm_8_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2|
<sequential_4/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp<sequential_4/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2z
;sequential_4/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp;sequential_4/lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2~
=sequential_4/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp=sequential_4/lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
Ä>
É
while_body_19543376
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	]G
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	]E
2while_lstm_cell_8_matmul_1_readvariableop_resource:	G@
1while_lstm_cell_8_biasadd_readvariableop_resource:	¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpÔ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMulÌ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp½
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMul_1´
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/addÅ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpÁ
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu°
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_1¥
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu_1´
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
ì%
ì
while_body_19540642
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_9_19540666_0:	G0
while_lstm_cell_9_19540668_0:
¢+
while_lstm_cell_9_19540670_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_9_19540666:	G.
while_lstm_cell_9_19540668:
¢)
while_lstm_cell_9_19540670:	¢)while/lstm_cell_9/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemæ
)while/lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_9_19540666_0while_lstm_cell_9_19540668_0while_lstm_cell_9_19540670_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_195406282+
)while/lstm_cell_9/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_9/StatefulPartitionedCall:output:0*
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
while/Identity_4Identity2while/lstm_cell_9/StatefulPartitionedCall:output:1^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_4¤
while/Identity_5Identity2while/lstm_cell_9/StatefulPartitionedCall:output:2^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_9_19540666while_lstm_cell_9_19540666_0":
while_lstm_cell_9_19540668while_lstm_cell_9_19540668_0":
while_lstm_cell_9_19540670while_lstm_cell_9_19540670_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2V
)while/lstm_cell_9/StatefulPartitionedCall)while/lstm_cell_9/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
ïK
²
!__inference__traced_save_19544564
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop8
4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableopB
>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop6
2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop8
4savev2_lstm_9_lstm_cell_9_kernel_read_readvariableopB
>savev2_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop6
2savev2_lstm_9_lstm_cell_9_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop?
;savev2_adam_lstm_8_lstm_cell_8_kernel_m_read_readvariableopI
Esavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_8_lstm_cell_8_bias_m_read_readvariableop?
;savev2_adam_lstm_9_lstm_cell_9_kernel_m_read_readvariableopI
Esavev2_adam_lstm_9_lstm_cell_9_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_9_lstm_cell_9_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop?
;savev2_adam_lstm_8_lstm_cell_8_kernel_v_read_readvariableopI
Esavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_8_lstm_cell_8_bias_v_read_readvariableop?
;savev2_adam_lstm_9_lstm_cell_9_kernel_v_read_readvariableopI
Esavev2_adam_lstm_9_lstm_cell_9_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_9_lstm_cell_9_bias_v_read_readvariableop
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
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop4savev2_lstm_8_lstm_cell_8_kernel_read_readvariableop>savev2_lstm_8_lstm_cell_8_recurrent_kernel_read_readvariableop2savev2_lstm_8_lstm_cell_8_bias_read_readvariableop4savev2_lstm_9_lstm_cell_9_kernel_read_readvariableop>savev2_lstm_9_lstm_cell_9_recurrent_kernel_read_readvariableop2savev2_lstm_9_lstm_cell_9_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop;savev2_adam_lstm_8_lstm_cell_8_kernel_m_read_readvariableopEsavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_8_lstm_cell_8_bias_m_read_readvariableop;savev2_adam_lstm_9_lstm_cell_9_kernel_m_read_readvariableopEsavev2_adam_lstm_9_lstm_cell_9_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_9_lstm_cell_9_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop;savev2_adam_lstm_8_lstm_cell_8_kernel_v_read_readvariableopEsavev2_adam_lstm_8_lstm_cell_8_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_8_lstm_cell_8_bias_v_read_readvariableop;savev2_adam_lstm_9_lstm_cell_9_kernel_v_read_readvariableopEsavev2_adam_lstm_9_lstm_cell_9_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_9_lstm_cell_9_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
ú: :	¢:: : : : : :	]:	G::	G:
¢:: : : : :	¢::	]:	G::	G:
¢::	¢::	]:	G::	G:
¢:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	¢: 
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
:	]:%	!

_output_shapes
:	G:!


_output_shapes	
::%!

_output_shapes
:	G:&"
 
_output_shapes
:
¢:!

_output_shapes	
::
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
:	¢: 

_output_shapes
::%!

_output_shapes
:	]:%!

_output_shapes
:	G:!

_output_shapes	
::%!

_output_shapes
:	G:&"
 
_output_shapes
:
¢:!

_output_shapes	
::%!

_output_shapes
:	¢: 

_output_shapes
::%!

_output_shapes
:	]:%!

_output_shapes
:	G:!

_output_shapes	
::%!

_output_shapes
:	G:& "
 
_output_shapes
:
¢:!!

_output_shapes	
::"

_output_shapes
: 
§

Ð
/__inference_sequential_4_layer_call_fn_19542067
lstm_8_input
unknown:	]
	unknown_0:	G
	unknown_1:	
	unknown_2:	G
	unknown_3:
¢
	unknown_4:	
	unknown_5:	¢
	unknown_6:
identity¢StatefulPartitionedCallÒ
StatefulPartitionedCallStatefulPartitionedCalllstm_8_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_195420272
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
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
&
_user_specified_namelstm_8_input
ß
Í
while_cond_19540011
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19540011___redundant_placeholder06
2while_while_cond_19540011___redundant_placeholder16
2while_while_cond_19540011___redundant_placeholder26
2while_while_cond_19540011___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
Ä>
É
while_body_19542923
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_8_matmul_readvariableop_resource_0:	]G
4while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GB
3while_lstm_cell_8_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_8_matmul_readvariableop_resource:	]E
2while_lstm_cell_8_matmul_1_readvariableop_resource:	G@
1while_lstm_cell_8_biasadd_readvariableop_resource:	¢(while/lstm_cell_8/BiasAdd/ReadVariableOp¢'while/lstm_cell_8/MatMul/ReadVariableOp¢)while/lstm_cell_8/MatMul_1/ReadVariableOpÃ
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
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype02)
'while/lstm_cell_8/MatMul/ReadVariableOpÔ
while/lstm_cell_8/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMulÌ
)while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype02+
)while/lstm_cell_8/MatMul_1/ReadVariableOp½
while/lstm_cell_8/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/MatMul_1´
while/lstm_cell_8/addAddV2"while/lstm_cell_8/MatMul:product:0$while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/addÅ
(while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_8/BiasAdd/ReadVariableOpÁ
while/lstm_cell_8/BiasAddBiasAddwhile/lstm_cell_8/add:z:00while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_8/BiasAdd
!while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_8/split/split_dim
while/lstm_cell_8/splitSplit*while/lstm_cell_8/split/split_dim:output:0"while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
while/lstm_cell_8/split
while/lstm_cell_8/SigmoidSigmoid while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid
while/lstm_cell_8/Sigmoid_1Sigmoid while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_1
while/lstm_cell_8/mulMulwhile/lstm_cell_8/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul
while/lstm_cell_8/ReluRelu while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu°
while/lstm_cell_8/mul_1Mulwhile/lstm_cell_8/Sigmoid:y:0$while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_1¥
while/lstm_cell_8/add_1AddV2while/lstm_cell_8/mul:z:0while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/add_1
while/lstm_cell_8/Sigmoid_2Sigmoid while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Sigmoid_2
while/lstm_cell_8/Relu_1Reluwhile/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/Relu_1´
while/lstm_cell_8/mul_2Mulwhile/lstm_cell_8/Sigmoid_2:y:0&while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/lstm_cell_8/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_8/mul_2:z:0*
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
while/Identity_3
while/Identity_4Identitywhile/lstm_cell_8/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_8/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_8/BiasAdd/ReadVariableOp(^while/lstm_cell_8/MatMul/ReadVariableOp*^while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_8_biasadd_readvariableop_resource3while_lstm_cell_8_biasadd_readvariableop_resource_0"j
2while_lstm_cell_8_matmul_1_readvariableop_resource4while_lstm_cell_8_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_8_matmul_readvariableop_resource2while_lstm_cell_8_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2T
(while/lstm_cell_8/BiasAdd/ReadVariableOp(while/lstm_cell_8/BiasAdd/ReadVariableOp2R
'while/lstm_cell_8/MatMul/ReadVariableOp'while/lstm_cell_8/MatMul/ReadVariableOp2V
)while/lstm_cell_8/MatMul_1/ReadVariableOp)while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
Ú>
Ë
while_body_19541421
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	GH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	GF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
¢@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOpÔ
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMulÍ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp½
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMul_1´
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/addÅ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOpÁ
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/BiasAdd
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dim
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
while/lstm_cell_9/split
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_1
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu±
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_1¦
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/add_1
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_2
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu_1µ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
Ô!
ý
E__inference_dense_4_layer_call_and_return_conditional_losses_19541551

inputs4
!tensordot_readvariableop_resource:	¢-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOp
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	¢*
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
:ÿÿÿÿÿÿÿÿÿ¢2
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
:ÿÿÿÿÿÿÿÿÿ¢: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
ã
Í
while_cond_19540641
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19540641___redundant_placeholder06
2while_while_cond_19540641___redundant_placeholder16
2while_while_cond_19540641___redundant_placeholder26
2while_while_cond_19540641___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:
¨F

D__inference_lstm_8_layer_call_and_return_conditional_losses_19540291

inputs'
lstm_cell_8_19540209:	]'
lstm_cell_8_19540211:	G#
lstm_cell_8_19540213:	
identity¢#lstm_cell_8/StatefulPartitionedCall¢whileD
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
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2	
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
strided_slice_2
#lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_8_19540209lstm_cell_8_19540211lstm_cell_8_19540213*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_195401442%
#lstm_cell_8/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2
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
while/loop_counterÇ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_8_19540209lstm_cell_8_19540211lstm_cell_8_19540213*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19540222*
condR
while_cond_19540221*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2

Identity|
NoOpNoOp$^lstm_cell_8/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2J
#lstm_cell_8/StatefulPartitionedCall#lstm_cell_8/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
¶[

D__inference_lstm_9_layer_call_and_return_conditional_losses_19544135

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	G@
,lstm_cell_9_matmul_1_readvariableop_resource:
¢:
+lstm_cell_9_biasadd_readvariableop_resource:	
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢whileD
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
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2	
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
:ÿÿÿÿÿÿÿÿÿG2
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
valueB"ÿÿÿÿG   27
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
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpª
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul¹
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOp¦
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul_1
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/add±
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOp©
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimó
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
lstm_cell_9/split
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_1
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_1
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/add_1
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu_1
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19544051*
condR
while_cond_19544050*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2

IdentityÅ
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ã
Í
while_cond_19543748
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19543748___redundant_placeholder06
2while_while_cond_19543748___redundant_placeholder16
2while_while_cond_19543748___redundant_placeholder26
2while_while_cond_19543748___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:
ã
Í
while_cond_19541689
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19541689___redundant_placeholder06
2while_while_cond_19541689___redundant_placeholder16
2while_while_cond_19541689___redundant_placeholder26
2while_while_cond_19541689___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:


I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_19544376

inputs
states_0
states_11
matmul_readvariableop_resource:	G4
 matmul_1_readvariableop_resource:
¢.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	G*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

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
A:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
"
_user_specified_name
states/1
Î
f
G__inference_dropout_8_layer_call_and_return_conditional_losses_19543521

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
:ÿÿÿÿÿÿÿÿÿG2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¸
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
dropout/Cast~
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
dropout/Mul_1i
IdentityIdentitydropout/Mul_1:z:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
Ò
H
,__inference_dropout_8_layer_call_fn_19543526

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_195413532
PartitionedCallp
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ß
Í
while_cond_19542922
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19542922___redundant_placeholder06
2while_while_cond_19542922___redundant_placeholder16
2while_while_cond_19542922___redundant_placeholder26
2while_while_cond_19542922___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
´
e
,__inference_dropout_8_layer_call_fn_19543531

inputs
identity¢StatefulPartitionedCallá
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_195418032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ß
Í
while_cond_19541255
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19541255___redundant_placeholder06
2while_while_cond_19541255___redundant_placeholder16
2while_while_cond_19541255___redundant_placeholder26
2while_while_cond_19541255___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
ºF

D__inference_lstm_9_layer_call_and_return_conditional_losses_19540711

inputs'
lstm_cell_9_19540629:	G(
lstm_cell_9_19540631:
¢#
lstm_cell_9_19540633:	
identity¢#lstm_cell_9/StatefulPartitionedCall¢whileD
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
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2	
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2
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
valueB"ÿÿÿÿG   27
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
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
strided_slice_2¢
#lstm_cell_9/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_9_19540629lstm_cell_9_19540631lstm_cell_9_19540633*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_195406282%
#lstm_cell_9/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2
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
while/loop_counterË
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_9_19540629lstm_cell_9_19540631lstm_cell_9_19540633*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19540642*
condR
while_cond_19540641*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢*
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2
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
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2

Identity|
NoOpNoOp$^lstm_cell_9/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG: : : 2J
#lstm_cell_9/StatefulPartitionedCall#lstm_cell_9/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
¹
e
,__inference_dropout_9_layer_call_fn_19544206

inputs
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_195416072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¢22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
ß
Í
while_cond_19543375
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19543375___redundant_placeholder06
2while_while_cond_19543375___redundant_placeholder16
2while_while_cond_19543375___redundant_placeholder26
2while_while_cond_19543375___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_19543073
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19543073___redundant_placeholder06
2while_while_cond_19543073___redundant_placeholder16
2while_while_cond_19543073___redundant_placeholder26
2while_while_cond_19543073___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
[
Ë
'sequential_4_lstm_9_while_body_19539811D
@sequential_4_lstm_9_while_sequential_4_lstm_9_while_loop_counterJ
Fsequential_4_lstm_9_while_sequential_4_lstm_9_while_maximum_iterations)
%sequential_4_lstm_9_while_placeholder+
'sequential_4_lstm_9_while_placeholder_1+
'sequential_4_lstm_9_while_placeholder_2+
'sequential_4_lstm_9_while_placeholder_3C
?sequential_4_lstm_9_while_sequential_4_lstm_9_strided_slice_1_0
{sequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	G\
Hsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢V
Gsequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	&
"sequential_4_lstm_9_while_identity(
$sequential_4_lstm_9_while_identity_1(
$sequential_4_lstm_9_while_identity_2(
$sequential_4_lstm_9_while_identity_3(
$sequential_4_lstm_9_while_identity_4(
$sequential_4_lstm_9_while_identity_5A
=sequential_4_lstm_9_while_sequential_4_lstm_9_strided_slice_1}
ysequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensorW
Dsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	GZ
Fsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:
¢T
Esequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	¢<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp¢;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp¢=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpë
Ksequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2M
Ksequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeË
=sequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_lstm_9_while_placeholderTsequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02?
=sequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItem
;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOpFsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype02=
;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp¤
,sequential_4/lstm_9/while/lstm_cell_9/MatMulMatMulDsequential_4/lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,sequential_4/lstm_9/while/lstm_cell_9/MatMul
=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOpHsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype02?
=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp
.sequential_4/lstm_9/while/lstm_cell_9/MatMul_1MatMul'sequential_4_lstm_9_while_placeholder_2Esequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ20
.sequential_4/lstm_9/while/lstm_cell_9/MatMul_1
)sequential_4/lstm_9/while/lstm_cell_9/addAddV26sequential_4/lstm_9/while/lstm_cell_9/MatMul:product:08sequential_4/lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)sequential_4/lstm_9/while/lstm_cell_9/add
<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOpGsequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02>
<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp
-sequential_4/lstm_9/while/lstm_cell_9/BiasAddBiasAdd-sequential_4/lstm_9/while/lstm_cell_9/add:z:0Dsequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_4/lstm_9/while/lstm_cell_9/BiasAdd°
5sequential_4/lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :27
5sequential_4/lstm_9/while/lstm_cell_9/split/split_dimÛ
+sequential_4/lstm_9/while/lstm_cell_9/splitSplit>sequential_4/lstm_9/while/lstm_cell_9/split/split_dim:output:06sequential_4/lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2-
+sequential_4/lstm_9/while/lstm_cell_9/splitÒ
-sequential_4/lstm_9/while/lstm_cell_9/SigmoidSigmoid4sequential_4/lstm_9/while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2/
-sequential_4/lstm_9/while/lstm_cell_9/SigmoidÖ
/sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid4sequential_4/lstm_9/while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢21
/sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_1î
)sequential_4/lstm_9/while/lstm_cell_9/mulMul3sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_1:y:0'sequential_4_lstm_9_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2+
)sequential_4/lstm_9/while/lstm_cell_9/mulÉ
*sequential_4/lstm_9/while/lstm_cell_9/ReluRelu4sequential_4/lstm_9/while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2,
*sequential_4/lstm_9/while/lstm_cell_9/Relu
+sequential_4/lstm_9/while/lstm_cell_9/mul_1Mul1sequential_4/lstm_9/while/lstm_cell_9/Sigmoid:y:08sequential_4/lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2-
+sequential_4/lstm_9/while/lstm_cell_9/mul_1ö
+sequential_4/lstm_9/while/lstm_cell_9/add_1AddV2-sequential_4/lstm_9/while/lstm_cell_9/mul:z:0/sequential_4/lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2-
+sequential_4/lstm_9/while/lstm_cell_9/add_1Ö
/sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid4sequential_4/lstm_9/while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢21
/sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_2È
,sequential_4/lstm_9/while/lstm_cell_9/Relu_1Relu/sequential_4/lstm_9/while/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2.
,sequential_4/lstm_9/while/lstm_cell_9/Relu_1
+sequential_4/lstm_9/while/lstm_cell_9/mul_2Mul3sequential_4/lstm_9/while/lstm_cell_9/Sigmoid_2:y:0:sequential_4/lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2-
+sequential_4/lstm_9/while/lstm_cell_9/mul_2Ã
>sequential_4/lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_lstm_9_while_placeholder_1%sequential_4_lstm_9_while_placeholder/sequential_4/lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype02@
>sequential_4/lstm_9/while/TensorArrayV2Write/TensorListSetItem
sequential_4/lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential_4/lstm_9/while/add/y¹
sequential_4/lstm_9/while/addAddV2%sequential_4_lstm_9_while_placeholder(sequential_4/lstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential_4/lstm_9/while/add
!sequential_4/lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!sequential_4/lstm_9/while/add_1/yÚ
sequential_4/lstm_9/while/add_1AddV2@sequential_4_lstm_9_while_sequential_4_lstm_9_while_loop_counter*sequential_4/lstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2!
sequential_4/lstm_9/while/add_1»
"sequential_4/lstm_9/while/IdentityIdentity#sequential_4/lstm_9/while/add_1:z:0^sequential_4/lstm_9/while/NoOp*
T0*
_output_shapes
: 2$
"sequential_4/lstm_9/while/Identityâ
$sequential_4/lstm_9/while/Identity_1IdentityFsequential_4_lstm_9_while_sequential_4_lstm_9_while_maximum_iterations^sequential_4/lstm_9/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_9/while/Identity_1½
$sequential_4/lstm_9/while/Identity_2Identity!sequential_4/lstm_9/while/add:z:0^sequential_4/lstm_9/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_9/while/Identity_2ê
$sequential_4/lstm_9/while/Identity_3IdentityNsequential_4/lstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/lstm_9/while/NoOp*
T0*
_output_shapes
: 2&
$sequential_4/lstm_9/while/Identity_3Ý
$sequential_4/lstm_9/while/Identity_4Identity/sequential_4/lstm_9/while/lstm_cell_9/mul_2:z:0^sequential_4/lstm_9/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2&
$sequential_4/lstm_9/while/Identity_4Ý
$sequential_4/lstm_9/while/Identity_5Identity/sequential_4/lstm_9/while/lstm_cell_9/add_1:z:0^sequential_4/lstm_9/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2&
$sequential_4/lstm_9/while/Identity_5¿
sequential_4/lstm_9/while/NoOpNoOp=^sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp<^sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp>^sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2 
sequential_4/lstm_9/while/NoOp"Q
"sequential_4_lstm_9_while_identity+sequential_4/lstm_9/while/Identity:output:0"U
$sequential_4_lstm_9_while_identity_1-sequential_4/lstm_9/while/Identity_1:output:0"U
$sequential_4_lstm_9_while_identity_2-sequential_4/lstm_9/while/Identity_2:output:0"U
$sequential_4_lstm_9_while_identity_3-sequential_4/lstm_9/while/Identity_3:output:0"U
$sequential_4_lstm_9_while_identity_4-sequential_4/lstm_9/while/Identity_4:output:0"U
$sequential_4_lstm_9_while_identity_5-sequential_4/lstm_9/while/Identity_5:output:0"
Esequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resourceGsequential_4_lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"
Fsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resourceHsequential_4_lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"
Dsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resourceFsequential_4_lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"
=sequential_4_lstm_9_while_sequential_4_lstm_9_strided_slice_1?sequential_4_lstm_9_while_sequential_4_lstm_9_strided_slice_1_0"ø
ysequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensor{sequential_4_lstm_9_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2|
<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp<sequential_4/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2z
;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp;sequential_4/lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2~
=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp=sequential_4/lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
Þ
¹
)__inference_lstm_9_layer_call_fn_19544146
inputs_0
unknown:	G
	unknown_0:
¢
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_9_layer_call_and_return_conditional_losses_195407112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
inputs/0


I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_19540628

inputs

states
states_11
matmul_readvariableop_resource:	G4
 matmul_1_readvariableop_resource:
¢.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	G*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
split`
SigmoidSigmoidsplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2	
Sigmoidd
	Sigmoid_1Sigmoidsplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
	Sigmoid_1]
mulMulSigmoid_1:y:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mulW
ReluRelusplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
Relui
mul_1MulSigmoid:y:0Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mul_1^
add_1AddV2mul:z:0	mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
add_1d
	Sigmoid_2Sigmoidsplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
	Sigmoid_2V
Relu_1Relu	add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
Relu_1m
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
mul_2e
IdentityIdentity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityi

Identity_1Identity	mul_2:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity_1i

Identity_2Identity	add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

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
A:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_namestates
[

D__inference_lstm_8_layer_call_and_return_conditional_losses_19543309

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	]?
,lstm_cell_8_matmul_1_readvariableop_resource:	G:
+lstm_cell_8_biasadd_readvariableop_resource:	
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢whileD
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
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2	
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
strided_slice_2²
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpª
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul¸
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp¦
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/add±
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp©
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimï
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19543225*
condR
while_cond_19543224*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG2
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
:ÿÿÿÿÿÿÿÿÿG2

IdentityÅ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
á%
ê
while_body_19540012
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0/
while_lstm_cell_8_19540036_0:	]/
while_lstm_cell_8_19540038_0:	G+
while_lstm_cell_8_19540040_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor-
while_lstm_cell_8_19540036:	]-
while_lstm_cell_8_19540038:	G)
while_lstm_cell_8_19540040:	¢)while/lstm_cell_8/StatefulPartitionedCallÃ
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
)while/TensorArrayV2Read/TensorListGetItemã
)while/lstm_cell_8/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_8_19540036_0while_lstm_cell_8_19540038_0while_lstm_cell_8_19540040_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_195399982+
)while/lstm_cell_8/StatefulPartitionedCallö
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder2while/lstm_cell_8/StatefulPartitionedCall:output:0*
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
while/Identity_3£
while/Identity_4Identity2while/lstm_cell_8/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_4£
while/Identity_5Identity2while/lstm_cell_8/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
while/Identity_5

while/NoOpNoOp*^while/lstm_cell_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0":
while_lstm_cell_8_19540036while_lstm_cell_8_19540036_0":
while_lstm_cell_8_19540038while_lstm_cell_8_19540038_0":
while_lstm_cell_8_19540040while_lstm_cell_8_19540040_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2V
)while/lstm_cell_8/StatefulPartitionedCall)while/lstm_cell_8/StatefulPartitionedCall: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
°
¶
)__inference_lstm_8_layer_call_fn_19543504

inputs
unknown:	]
	unknown_0:	G
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_8_layer_call_and_return_conditional_losses_195419702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

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
[

D__inference_lstm_8_layer_call_and_return_conditional_losses_19543460

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	]?
,lstm_cell_8_matmul_1_readvariableop_resource:	G:
+lstm_cell_8_biasadd_readvariableop_resource:	
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢whileD
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
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2	
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
strided_slice_2²
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpª
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul¸
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp¦
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/add±
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp©
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimï
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19543376*
condR
while_cond_19543375*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG2
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
:ÿÿÿÿÿÿÿÿÿG2

IdentityÅ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
´
·
)__inference_lstm_9_layer_call_fn_19544168

inputs
unknown:	G
	unknown_0:
¢
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_9_layer_call_and_return_conditional_losses_195415052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ã
Í
while_cond_19540851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19540851___redundant_placeholder06
2while_while_cond_19540851___redundant_placeholder16
2while_while_cond_19540851___redundant_placeholder26
2while_while_cond_19540851___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:
ÙH
«

lstm_9_while_body_19542361*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3)
%lstm_9_while_lstm_9_strided_slice_1_0e
alstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0:	GO
;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢I
:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0:	
lstm_9_while_identity
lstm_9_while_identity_1
lstm_9_while_identity_2
lstm_9_while_identity_3
lstm_9_while_identity_4
lstm_9_while_identity_5'
#lstm_9_while_lstm_9_strided_slice_1c
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensorJ
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource:	GM
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource:
¢G
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:	¢/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp¢.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp¢0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpÑ
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2@
>lstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shapeý
0lstm_9/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0lstm_9_while_placeholderGlstm_9/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype022
0lstm_9/while/TensorArrayV2Read/TensorListGetItemÛ
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype020
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOpð
lstm_9/while/lstm_cell_9/MatMulMatMul7lstm_9/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_9/while/lstm_cell_9/MatMulâ
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype022
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOpÙ
!lstm_9/while/lstm_cell_9/MatMul_1MatMullstm_9_while_placeholder_28lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_9/while/lstm_cell_9/MatMul_1Ð
lstm_9/while/lstm_cell_9/addAddV2)lstm_9/while/lstm_cell_9/MatMul:product:0+lstm_9/while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/while/lstm_cell_9/addÚ
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOpÝ
 lstm_9/while/lstm_cell_9/BiasAddBiasAdd lstm_9/while/lstm_cell_9/add:z:07lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_9/while/lstm_cell_9/BiasAdd
(lstm_9/while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_9/while/lstm_cell_9/split/split_dim§
lstm_9/while/lstm_cell_9/splitSplit1lstm_9/while/lstm_cell_9/split/split_dim:output:0)lstm_9/while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2 
lstm_9/while/lstm_cell_9/split«
 lstm_9/while/lstm_cell_9/SigmoidSigmoid'lstm_9/while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2"
 lstm_9/while/lstm_cell_9/Sigmoid¯
"lstm_9/while/lstm_cell_9/Sigmoid_1Sigmoid'lstm_9/while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2$
"lstm_9/while/lstm_cell_9/Sigmoid_1º
lstm_9/while/lstm_cell_9/mulMul&lstm_9/while/lstm_cell_9/Sigmoid_1:y:0lstm_9_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/while/lstm_cell_9/mul¢
lstm_9/while/lstm_cell_9/ReluRelu'lstm_9/while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/while/lstm_cell_9/ReluÍ
lstm_9/while/lstm_cell_9/mul_1Mul$lstm_9/while/lstm_cell_9/Sigmoid:y:0+lstm_9/while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2 
lstm_9/while/lstm_cell_9/mul_1Â
lstm_9/while/lstm_cell_9/add_1AddV2 lstm_9/while/lstm_cell_9/mul:z:0"lstm_9/while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2 
lstm_9/while/lstm_cell_9/add_1¯
"lstm_9/while/lstm_cell_9/Sigmoid_2Sigmoid'lstm_9/while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2$
"lstm_9/while/lstm_cell_9/Sigmoid_2¡
lstm_9/while/lstm_cell_9/Relu_1Relu"lstm_9/while/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2!
lstm_9/while/lstm_cell_9/Relu_1Ñ
lstm_9/while/lstm_cell_9/mul_2Mul&lstm_9/while/lstm_cell_9/Sigmoid_2:y:0-lstm_9/while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2 
lstm_9/while/lstm_cell_9/mul_2
1lstm_9/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_9_while_placeholder_1lstm_9_while_placeholder"lstm_9/while/lstm_cell_9/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_9/while/TensorArrayV2Write/TensorListSetItemj
lstm_9/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add/y
lstm_9/while/addAddV2lstm_9_while_placeholderlstm_9/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/addn
lstm_9/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_9/while/add_1/y
lstm_9/while/add_1AddV2&lstm_9_while_lstm_9_while_loop_counterlstm_9/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_9/while/add_1
lstm_9/while/IdentityIdentitylstm_9/while/add_1:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity¡
lstm_9/while/Identity_1Identity,lstm_9_while_lstm_9_while_maximum_iterations^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_1
lstm_9/while/Identity_2Identitylstm_9/while/add:z:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_2¶
lstm_9/while/Identity_3IdentityAlstm_9/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_9/while/NoOp*
T0*
_output_shapes
: 2
lstm_9/while/Identity_3©
lstm_9/while/Identity_4Identity"lstm_9/while/lstm_cell_9/mul_2:z:0^lstm_9/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/while/Identity_4©
lstm_9/while/Identity_5Identity"lstm_9/while/lstm_cell_9/add_1:z:0^lstm_9/while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/while/Identity_5þ
lstm_9/while/NoOpNoOp0^lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/^lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp1^lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_9/while/NoOp"7
lstm_9_while_identitylstm_9/while/Identity:output:0";
lstm_9_while_identity_1 lstm_9/while/Identity_1:output:0";
lstm_9_while_identity_2 lstm_9/while/Identity_2:output:0";
lstm_9_while_identity_3 lstm_9/while/Identity_3:output:0";
lstm_9_while_identity_4 lstm_9/while/Identity_4:output:0";
lstm_9_while_identity_5 lstm_9/while/Identity_5:output:0"L
#lstm_9_while_lstm_9_strided_slice_1%lstm_9_while_lstm_9_strided_slice_1_0"v
8lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource:lstm_9_while_lstm_cell_9_biasadd_readvariableop_resource_0"x
9lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource;lstm_9_while_lstm_cell_9_matmul_1_readvariableop_resource_0"t
7lstm_9_while_lstm_cell_9_matmul_readvariableop_resource9lstm_9_while_lstm_cell_9_matmul_readvariableop_resource_0"Ä
_lstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensoralstm_9_while_tensorarrayv2read_tensorlistgetitem_lstm_9_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2b
/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp/lstm_9/while/lstm_cell_9/BiasAdd/ReadVariableOp2`
.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp.lstm_9/while/lstm_cell_9/MatMul/ReadVariableOp2d
0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp0lstm_9/while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
´
·
)__inference_lstm_9_layer_call_fn_19544179

inputs
unknown:	G
	unknown_0:
¢
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_9_layer_call_and_return_conditional_losses_195417742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ü

I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_19544278

inputs
states_0
states_11
matmul_readvariableop_resource:	]3
 matmul_1_readvariableop_resource:	G.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
states/1
Ú>
Ë
while_body_19543749
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_9_matmul_readvariableop_resource_0:	GH
4while_lstm_cell_9_matmul_1_readvariableop_resource_0:
¢B
3while_lstm_cell_9_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_9_matmul_readvariableop_resource:	GF
2while_lstm_cell_9_matmul_1_readvariableop_resource:
¢@
1while_lstm_cell_9_biasadd_readvariableop_resource:	¢(while/lstm_cell_9/BiasAdd/ReadVariableOp¢'while/lstm_cell_9/MatMul/ReadVariableOp¢)while/lstm_cell_9/MatMul_1/ReadVariableOpÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÓ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÆ
'while/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_9_matmul_readvariableop_resource_0*
_output_shapes
:	G*
dtype02)
'while/lstm_cell_9/MatMul/ReadVariableOpÔ
while/lstm_cell_9/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMulÍ
)while/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_9_matmul_1_readvariableop_resource_0* 
_output_shapes
:
¢*
dtype02+
)while/lstm_cell_9/MatMul_1/ReadVariableOp½
while/lstm_cell_9/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/MatMul_1´
while/lstm_cell_9/addAddV2"while/lstm_cell_9/MatMul:product:0$while/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/addÅ
(while/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_9_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype02*
(while/lstm_cell_9/BiasAdd/ReadVariableOpÁ
while/lstm_cell_9/BiasAddBiasAddwhile/lstm_cell_9/add:z:00while/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
while/lstm_cell_9/BiasAdd
!while/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2#
!while/lstm_cell_9/split/split_dim
while/lstm_cell_9/splitSplit*while/lstm_cell_9/split/split_dim:output:0"while/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
while/lstm_cell_9/split
while/lstm_cell_9/SigmoidSigmoid while/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid
while/lstm_cell_9/Sigmoid_1Sigmoid while/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_1
while/lstm_cell_9/mulMulwhile/lstm_cell_9/Sigmoid_1:y:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul
while/lstm_cell_9/ReluRelu while/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu±
while/lstm_cell_9/mul_1Mulwhile/lstm_cell_9/Sigmoid:y:0$while/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_1¦
while/lstm_cell_9/add_1AddV2while/lstm_cell_9/mul:z:0while/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/add_1
while/lstm_cell_9/Sigmoid_2Sigmoid while/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Sigmoid_2
while/lstm_cell_9/Relu_1Reluwhile/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/Relu_1µ
while/lstm_cell_9/mul_2Mulwhile/lstm_cell_9/Sigmoid_2:y:0&while/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/lstm_cell_9/mul_2ß
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_9/mul_2:z:0*
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
while/Identity_4Identitywhile/lstm_cell_9/mul_2:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_4
while/Identity_5Identitywhile/lstm_cell_9/add_1:z:0^while/NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
while/Identity_5Û

while/NoOpNoOp)^while/lstm_cell_9/BiasAdd/ReadVariableOp(^while/lstm_cell_9/MatMul/ReadVariableOp*^while/lstm_cell_9/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2

while/NoOp")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_9_biasadd_readvariableop_resource3while_lstm_cell_9_biasadd_readvariableop_resource_0"j
2while_lstm_cell_9_matmul_1_readvariableop_resource4while_lstm_cell_9_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_9_matmul_readvariableop_resource2while_lstm_cell_9_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : 2T
(while/lstm_cell_9/BiasAdd/ReadVariableOp(while/lstm_cell_9/BiasAdd/ReadVariableOp2R
'while/lstm_cell_9/MatMul/ReadVariableOp'while/lstm_cell_9/MatMul/ReadVariableOp2V
)while/lstm_cell_9/MatMul_1/ReadVariableOp)while/lstm_cell_9/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
: 
¶[

D__inference_lstm_9_layer_call_and_return_conditional_losses_19541505

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	G@
,lstm_cell_9_matmul_1_readvariableop_resource:
¢:
+lstm_cell_9_biasadd_readvariableop_resource:	
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢whileD
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
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2	
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
:ÿÿÿÿÿÿÿÿÿG2
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
valueB"ÿÿÿÿG   27
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
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpª
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul¹
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOp¦
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul_1
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/add±
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOp©
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimó
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
lstm_cell_9/split
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_1
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_1
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/add_1
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu_1
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19541421*
condR
while_cond_19541420*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2

IdentityÅ
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs


Ê
/__inference_sequential_4_layer_call_fn_19542856

inputs
unknown:	]
	unknown_0:	G
	unknown_1:	
	unknown_2:	G
	unknown_3:
¢
	unknown_4:	
	unknown_5:	¢
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_195420272
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
¸
÷
.__inference_lstm_cell_8_layer_call_fn_19544344

inputs
states_0
states_1
unknown:	]
	unknown_0:	G
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallÄ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_195401442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
states/1
¿
ë
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542814

inputsD
1lstm_8_lstm_cell_8_matmul_readvariableop_resource:	]F
3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource:	GA
2lstm_8_lstm_cell_8_biasadd_readvariableop_resource:	D
1lstm_9_lstm_cell_9_matmul_readvariableop_resource:	GG
3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource:
¢A
2lstm_9_lstm_cell_9_biasadd_readvariableop_resource:	<
)dense_4_tensordot_readvariableop_resource:	¢5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_4/BiasAdd/ReadVariableOp¢ dense_4/Tensordot/ReadVariableOp¢)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp¢(lstm_8/lstm_cell_8/MatMul/ReadVariableOp¢*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp¢lstm_8/while¢)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp¢(lstm_9/lstm_cell_9/MatMul/ReadVariableOp¢*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp¢lstm_9/whileR
lstm_8/ShapeShapeinputs*
T0*
_output_shapes
:2
lstm_8/Shape
lstm_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice/stack
lstm_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_1
lstm_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_8/strided_slice/stack_2
lstm_8/strided_sliceStridedSlicelstm_8/Shape:output:0#lstm_8/strided_slice/stack:output:0%lstm_8/strided_slice/stack_1:output:0%lstm_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slicej
lstm_8/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
lstm_8/zeros/mul/y
lstm_8/zeros/mulMullstm_8/strided_slice:output:0lstm_8/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/mulm
lstm_8/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_8/zeros/Less/y
lstm_8/zeros/LessLesslstm_8/zeros/mul:z:0lstm_8/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros/Lessp
lstm_8/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2
lstm_8/zeros/packed/1
lstm_8/zeros/packedPacklstm_8/strided_slice:output:0lstm_8/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros/packedm
lstm_8/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros/Const
lstm_8/zerosFilllstm_8/zeros/packed:output:0lstm_8/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/zerosn
lstm_8/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
lstm_8/zeros_1/mul/y
lstm_8/zeros_1/mulMullstm_8/strided_slice:output:0lstm_8/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/mulq
lstm_8/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_8/zeros_1/Less/y
lstm_8/zeros_1/LessLesslstm_8/zeros_1/mul:z:0lstm_8/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_8/zeros_1/Lesst
lstm_8/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :G2
lstm_8/zeros_1/packed/1¥
lstm_8/zeros_1/packedPacklstm_8/strided_slice:output:0 lstm_8/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_8/zeros_1/packedq
lstm_8/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/zeros_1/Const
lstm_8/zeros_1Filllstm_8/zeros_1/packed:output:0lstm_8/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/zeros_1
lstm_8/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose/perm
lstm_8/transpose	Transposeinputslstm_8/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]2
lstm_8/transposed
lstm_8/Shape_1Shapelstm_8/transpose:y:0*
T0*
_output_shapes
:2
lstm_8/Shape_1
lstm_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_1/stack
lstm_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_1
lstm_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_1/stack_2
lstm_8/strided_slice_1StridedSlicelstm_8/Shape_1:output:0%lstm_8/strided_slice_1/stack:output:0'lstm_8/strided_slice_1/stack_1:output:0'lstm_8/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_8/strided_slice_1
"lstm_8/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_8/TensorArrayV2/element_shapeÎ
lstm_8/TensorArrayV2TensorListReserve+lstm_8/TensorArrayV2/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2Í
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2>
<lstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_8/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_8/transpose:y:0Elstm_8/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_8/TensorArrayUnstack/TensorListFromTensor
lstm_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_8/strided_slice_2/stack
lstm_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_1
lstm_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_2/stack_2¦
lstm_8/strided_slice_2StridedSlicelstm_8/transpose:y:0%lstm_8/strided_slice_2/stack:output:0'lstm_8/strided_slice_2/stack_1:output:0'lstm_8/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
shrink_axis_mask2
lstm_8/strided_slice_2Ç
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp1lstm_8_lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02*
(lstm_8/lstm_cell_8/MatMul/ReadVariableOpÆ
lstm_8/lstm_cell_8/MatMulMatMullstm_8/strided_slice_2:output:00lstm_8/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/lstm_cell_8/MatMulÍ
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02,
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOpÂ
lstm_8/lstm_cell_8/MatMul_1MatMullstm_8/zeros:output:02lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/lstm_cell_8/MatMul_1¸
lstm_8/lstm_cell_8/addAddV2#lstm_8/lstm_cell_8/MatMul:product:0%lstm_8/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/lstm_cell_8/addÆ
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOpÅ
lstm_8/lstm_cell_8/BiasAddBiasAddlstm_8/lstm_cell_8/add:z:01lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/lstm_cell_8/BiasAdd
"lstm_8/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_8/lstm_cell_8/split/split_dim
lstm_8/lstm_cell_8/splitSplit+lstm_8/lstm_cell_8/split/split_dim:output:0#lstm_8/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
lstm_8/lstm_cell_8/split
lstm_8/lstm_cell_8/SigmoidSigmoid!lstm_8/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Sigmoid
lstm_8/lstm_cell_8/Sigmoid_1Sigmoid!lstm_8/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Sigmoid_1¤
lstm_8/lstm_cell_8/mulMul lstm_8/lstm_cell_8/Sigmoid_1:y:0lstm_8/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/mul
lstm_8/lstm_cell_8/ReluRelu!lstm_8/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Relu´
lstm_8/lstm_cell_8/mul_1Mullstm_8/lstm_cell_8/Sigmoid:y:0%lstm_8/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/mul_1©
lstm_8/lstm_cell_8/add_1AddV2lstm_8/lstm_cell_8/mul:z:0lstm_8/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/add_1
lstm_8/lstm_cell_8/Sigmoid_2Sigmoid!lstm_8/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Sigmoid_2
lstm_8/lstm_cell_8/Relu_1Relulstm_8/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/Relu_1¸
lstm_8/lstm_cell_8/mul_2Mul lstm_8/lstm_cell_8/Sigmoid_2:y:0'lstm_8/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/lstm_cell_8/mul_2
$lstm_8/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2&
$lstm_8/TensorArrayV2_1/element_shapeÔ
lstm_8/TensorArrayV2_1TensorListReserve-lstm_8/TensorArrayV2_1/element_shape:output:0lstm_8/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_8/TensorArrayV2_1\
lstm_8/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/time
lstm_8/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_8/while/maximum_iterationsx
lstm_8/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_8/while/loop_counterõ
lstm_8/whileWhile"lstm_8/while/loop_counter:output:0(lstm_8/while/maximum_iterations:output:0lstm_8/time:output:0lstm_8/TensorArrayV2_1:handle:0lstm_8/zeros:output:0lstm_8/zeros_1:output:0lstm_8/strided_slice_1:output:0>lstm_8/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_8_lstm_cell_8_matmul_readvariableop_resource3lstm_8_lstm_cell_8_matmul_1_readvariableop_resource2lstm_8_lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_8_while_body_19542540*&
condR
lstm_8_while_cond_19542539*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
lstm_8/whileÃ
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   29
7lstm_8/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_8/TensorArrayV2Stack/TensorListStackTensorListStacklstm_8/while:output:3@lstm_8/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
element_dtype02+
)lstm_8/TensorArrayV2Stack/TensorListStack
lstm_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_8/strided_slice_3/stack
lstm_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_8/strided_slice_3/stack_1
lstm_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_8/strided_slice_3/stack_2Ä
lstm_8/strided_slice_3StridedSlice2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_8/strided_slice_3/stack:output:0'lstm_8/strided_slice_3/stack_1:output:0'lstm_8/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
lstm_8/strided_slice_3
lstm_8/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_8/transpose_1/permÁ
lstm_8/transpose_1	Transpose2lstm_8/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_8/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/transpose_1t
lstm_8/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_8/runtimew
dropout_8/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *«ªª?2
dropout_8/dropout/Const¥
dropout_8/dropout/MulMullstm_8/transpose_1:y:0 dropout_8/dropout/Const:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
dropout_8/dropout/Mulx
dropout_8/dropout/ShapeShapelstm_8/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_8/dropout/ShapeÖ
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
dtype020
.dropout_8/dropout/random_uniform/RandomUniform
 dropout_8/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  >2"
 dropout_8/dropout/GreaterEqual/yê
dropout_8/dropout/GreaterEqualGreaterEqual7dropout_8/dropout/random_uniform/RandomUniform:output:0)dropout_8/dropout/GreaterEqual/y:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2 
dropout_8/dropout/GreaterEqual¡
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
dropout_8/dropout/Cast¦
dropout_8/dropout/Mul_1Muldropout_8/dropout/Mul:z:0dropout_8/dropout/Cast:y:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
dropout_8/dropout/Mul_1g
lstm_9/ShapeShapedropout_8/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
lstm_9/Shape
lstm_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice/stack
lstm_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_1
lstm_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm_9/strided_slice/stack_2
lstm_9/strided_sliceStridedSlicelstm_9/Shape:output:0#lstm_9/strided_slice/stack:output:0%lstm_9/strided_slice/stack_1:output:0%lstm_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slicek
lstm_9/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
lstm_9/zeros/mul/y
lstm_9/zeros/mulMullstm_9/strided_slice:output:0lstm_9/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/mulm
lstm_9/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_9/zeros/Less/y
lstm_9/zeros/LessLesslstm_9/zeros/mul:z:0lstm_9/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros/Lessq
lstm_9/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¢2
lstm_9/zeros/packed/1
lstm_9/zeros/packedPacklstm_9/strided_slice:output:0lstm_9/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros/packedm
lstm_9/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros/Const
lstm_9/zerosFilllstm_9/zeros/packed:output:0lstm_9/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/zeroso
lstm_9/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
lstm_9/zeros_1/mul/y
lstm_9/zeros_1/mulMullstm_9/strided_slice:output:0lstm_9/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/mulq
lstm_9/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm_9/zeros_1/Less/y
lstm_9/zeros_1/LessLesslstm_9/zeros_1/mul:z:0lstm_9/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm_9/zeros_1/Lessu
lstm_9/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :¢2
lstm_9/zeros_1/packed/1¥
lstm_9/zeros_1/packedPacklstm_9/strided_slice:output:0 lstm_9/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm_9/zeros_1/packedq
lstm_9/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/zeros_1/Const
lstm_9/zeros_1Filllstm_9/zeros_1/packed:output:0lstm_9/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/zeros_1
lstm_9/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose/perm¤
lstm_9/transpose	Transposedropout_8/dropout/Mul_1:z:0lstm_9/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_9/transposed
lstm_9/Shape_1Shapelstm_9/transpose:y:0*
T0*
_output_shapes
:2
lstm_9/Shape_1
lstm_9/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_1/stack
lstm_9/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_1
lstm_9/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_1/stack_2
lstm_9/strided_slice_1StridedSlicelstm_9/Shape_1:output:0%lstm_9/strided_slice_1/stack:output:0'lstm_9/strided_slice_1/stack_1:output:0'lstm_9/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm_9/strided_slice_1
"lstm_9/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2$
"lstm_9/TensorArrayV2/element_shapeÎ
lstm_9/TensorArrayV2TensorListReserve+lstm_9/TensorArrayV2/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2Í
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2>
<lstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape
.lstm_9/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_9/transpose:y:0Elstm_9/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type020
.lstm_9/TensorArrayUnstack/TensorListFromTensor
lstm_9/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm_9/strided_slice_2/stack
lstm_9/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_1
lstm_9/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_2/stack_2¦
lstm_9/strided_slice_2StridedSlicelstm_9/transpose:y:0%lstm_9/strided_slice_2/stack:output:0'lstm_9/strided_slice_2/stack_1:output:0'lstm_9/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
lstm_9/strided_slice_2Ç
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpReadVariableOp1lstm_9_lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype02*
(lstm_9/lstm_cell_9/MatMul/ReadVariableOpÆ
lstm_9/lstm_cell_9/MatMulMatMullstm_9/strided_slice_2:output:00lstm_9/lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/lstm_cell_9/MatMulÎ
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02,
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOpÂ
lstm_9/lstm_cell_9/MatMul_1MatMullstm_9/zeros:output:02lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/lstm_cell_9/MatMul_1¸
lstm_9/lstm_cell_9/addAddV2#lstm_9/lstm_cell_9/MatMul:product:0%lstm_9/lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/lstm_cell_9/addÆ
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOpÅ
lstm_9/lstm_cell_9/BiasAddBiasAddlstm_9/lstm_cell_9/add:z:01lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_9/lstm_cell_9/BiasAdd
"lstm_9/lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2$
"lstm_9/lstm_cell_9/split/split_dim
lstm_9/lstm_cell_9/splitSplit+lstm_9/lstm_cell_9/split/split_dim:output:0#lstm_9/lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
lstm_9/lstm_cell_9/split
lstm_9/lstm_cell_9/SigmoidSigmoid!lstm_9/lstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Sigmoid
lstm_9/lstm_cell_9/Sigmoid_1Sigmoid!lstm_9/lstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Sigmoid_1¥
lstm_9/lstm_cell_9/mulMul lstm_9/lstm_cell_9/Sigmoid_1:y:0lstm_9/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/mul
lstm_9/lstm_cell_9/ReluRelu!lstm_9/lstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Reluµ
lstm_9/lstm_cell_9/mul_1Mullstm_9/lstm_cell_9/Sigmoid:y:0%lstm_9/lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/mul_1ª
lstm_9/lstm_cell_9/add_1AddV2lstm_9/lstm_cell_9/mul:z:0lstm_9/lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/add_1
lstm_9/lstm_cell_9/Sigmoid_2Sigmoid!lstm_9/lstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Sigmoid_2
lstm_9/lstm_cell_9/Relu_1Relulstm_9/lstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/Relu_1¹
lstm_9/lstm_cell_9/mul_2Mul lstm_9/lstm_cell_9/Sigmoid_2:y:0'lstm_9/lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/lstm_cell_9/mul_2
$lstm_9/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2&
$lstm_9/TensorArrayV2_1/element_shapeÔ
lstm_9/TensorArrayV2_1TensorListReserve-lstm_9/TensorArrayV2_1/element_shape:output:0lstm_9/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm_9/TensorArrayV2_1\
lstm_9/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/time
lstm_9/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
lstm_9/while/maximum_iterationsx
lstm_9/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_9/while/loop_counterù
lstm_9/whileWhile"lstm_9/while/loop_counter:output:0(lstm_9/while/maximum_iterations:output:0lstm_9/time:output:0lstm_9/TensorArrayV2_1:handle:0lstm_9/zeros:output:0lstm_9/zeros_1:output:0lstm_9/strided_slice_1:output:0>lstm_9/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_9_lstm_cell_9_matmul_readvariableop_resource3lstm_9_lstm_cell_9_matmul_1_readvariableop_resource2lstm_9_lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *&
bodyR
lstm_9_while_body_19542695*&
condR
lstm_9_while_cond_19542694*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
lstm_9/whileÃ
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   29
7lstm_9/TensorArrayV2Stack/TensorListStack/element_shape
)lstm_9/TensorArrayV2Stack/TensorListStackTensorListStacklstm_9/while:output:3@lstm_9/TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
element_dtype02+
)lstm_9/TensorArrayV2Stack/TensorListStack
lstm_9/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm_9/strided_slice_3/stack
lstm_9/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2 
lstm_9/strided_slice_3/stack_1
lstm_9/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
lstm_9/strided_slice_3/stack_2Å
lstm_9/strided_slice_3StridedSlice2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_9/strided_slice_3/stack:output:0'lstm_9/strided_slice_3/stack_1:output:0'lstm_9/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
shrink_axis_mask2
lstm_9/strided_slice_3
lstm_9/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm_9/transpose_1/permÂ
lstm_9/transpose_1	Transpose2lstm_9/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_9/transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_9/transpose_1t
lstm_9/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_9/runtimew
dropout_9/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout_9/dropout/Const¦
dropout_9/dropout/MulMullstm_9/transpose_1:y:0 dropout_9/dropout/Const:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dropout_9/dropout/Mulx
dropout_9/dropout/ShapeShapelstm_9/transpose_1:y:0*
T0*
_output_shapes
:2
dropout_9/dropout/Shape×
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
dtype020
.dropout_9/dropout/random_uniform/RandomUniform
 dropout_9/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2"
 dropout_9/dropout/GreaterEqual/yë
dropout_9/dropout/GreaterEqualGreaterEqual7dropout_9/dropout/random_uniform/RandomUniform:output:0)dropout_9/dropout/GreaterEqual/y:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2 
dropout_9/dropout/GreaterEqual¢
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dropout_9/dropout/Cast§
dropout_9/dropout/Mul_1Muldropout_9/dropout/Mul:z:0dropout_9/dropout/Cast:y:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dropout_9/dropout/Mul_1¯
 dense_4/Tensordot/ReadVariableOpReadVariableOp)dense_4_tensordot_readvariableop_resource*
_output_shapes
:	¢*
dtype02"
 dense_4/Tensordot/ReadVariableOpz
dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/axes
dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_4/Tensordot/free}
dense_4/Tensordot/ShapeShapedropout_9/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_4/Tensordot/Shape
dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/GatherV2/axisù
dense_4/Tensordot/GatherV2GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/free:output:0(dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2
!dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_4/Tensordot/GatherV2_1/axisÿ
dense_4/Tensordot/GatherV2_1GatherV2 dense_4/Tensordot/Shape:output:0dense_4/Tensordot/axes:output:0*dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_4/Tensordot/GatherV2_1|
dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const 
dense_4/Tensordot/ProdProd#dense_4/Tensordot/GatherV2:output:0 dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod
dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_4/Tensordot/Const_1¨
dense_4/Tensordot/Prod_1Prod%dense_4/Tensordot/GatherV2_1:output:0"dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_4/Tensordot/Prod_1
dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_4/Tensordot/concat/axisØ
dense_4/Tensordot/concatConcatV2dense_4/Tensordot/free:output:0dense_4/Tensordot/axes:output:0&dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat¬
dense_4/Tensordot/stackPackdense_4/Tensordot/Prod:output:0!dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/stack¾
dense_4/Tensordot/transpose	Transposedropout_9/dropout/Mul_1:z:0!dense_4/Tensordot/concat:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
dense_4/Tensordot/transpose¿
dense_4/Tensordot/ReshapeReshapedense_4/Tensordot/transpose:y:0 dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_4/Tensordot/Reshape¾
dense_4/Tensordot/MatMulMatMul"dense_4/Tensordot/Reshape:output:0(dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Tensordot/MatMul
dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_4/Tensordot/Const_2
dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_4/Tensordot/concat_1/axiså
dense_4/Tensordot/concat_1ConcatV2#dense_4/Tensordot/GatherV2:output:0"dense_4/Tensordot/Const_2:output:0(dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_4/Tensordot/concat_1°
dense_4/TensordotReshape"dense_4/Tensordot/MatMul:product:0#dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Tensordot¤
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp§
dense_4/BiasAddBiasAdddense_4/Tensordot:output:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAdd}
dense_4/SoftmaxSoftmaxdense_4/BiasAdd:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Softmaxx
IdentityIdentitydense_4/Softmax:softmax:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity¸
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp!^dense_4/Tensordot/ReadVariableOp*^lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)^lstm_8/lstm_cell_8/MatMul/ReadVariableOp+^lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp^lstm_8/while*^lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)^lstm_9/lstm_cell_9/MatMul/ReadVariableOp+^lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp^lstm_9/while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2D
 dense_4/Tensordot/ReadVariableOp dense_4/Tensordot/ReadVariableOp2V
)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp)lstm_8/lstm_cell_8/BiasAdd/ReadVariableOp2T
(lstm_8/lstm_cell_8/MatMul/ReadVariableOp(lstm_8/lstm_cell_8/MatMul/ReadVariableOp2X
*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp*lstm_8/lstm_cell_8/MatMul_1/ReadVariableOp2
lstm_8/whilelstm_8/while2V
)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp)lstm_9/lstm_cell_9/BiasAdd/ReadVariableOp2T
(lstm_9/lstm_cell_9/MatMul/ReadVariableOp(lstm_9/lstm_cell_9/MatMul/ReadVariableOp2X
*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp*lstm_9/lstm_cell_9/MatMul_1/ReadVariableOp2
lstm_9/whilelstm_9/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
Ø[

D__inference_lstm_8_layer_call_and_return_conditional_losses_19543158
inputs_0=
*lstm_cell_8_matmul_readvariableop_resource:	]?
,lstm_cell_8_matmul_1_readvariableop_resource:	G:
+lstm_cell_8_biasadd_readvariableop_resource:	
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢whileF
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
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2	
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
strided_slice_2²
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpª
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul¸
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp¦
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/add±
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp©
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimï
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19543074*
condR
while_cond_19543073*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   22
0TensorArrayV2Stack/TensorListStack/element_shapeñ
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG*
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2
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
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG2

IdentityÅ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]
"
_user_specified_name
inputs/0
÷
«
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542027

inputs"
lstm_8_19542005:	]"
lstm_8_19542007:	G
lstm_8_19542009:	"
lstm_9_19542013:	G#
lstm_9_19542015:
¢
lstm_9_19542017:	#
dense_4_19542021:	¢
dense_4_19542023:
identity¢dense_4/StatefulPartitionedCall¢!dropout_8/StatefulPartitionedCall¢!dropout_9/StatefulPartitionedCall¢lstm_8/StatefulPartitionedCall¢lstm_9/StatefulPartitionedCall§
lstm_8/StatefulPartitionedCallStatefulPartitionedCallinputslstm_8_19542005lstm_8_19542007lstm_8_19542009*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_8_layer_call_and_return_conditional_losses_195419702 
lstm_8/StatefulPartitionedCall
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall'lstm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_8_layer_call_and_return_conditional_losses_195418032#
!dropout_8/StatefulPartitionedCallÌ
lstm_9/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0lstm_9_19542013lstm_9_19542015lstm_9_19542017*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_9_layer_call_and_return_conditional_losses_195417742 
lstm_9/StatefulPartitionedCall»
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall'lstm_9/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_195416072#
!dropout_9/StatefulPartitionedCall½
dense_4/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0dense_4_19542021dense_4_19542023*
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
E__inference_dense_4_layer_call_and_return_conditional_losses_195415512!
dense_4/StatefulPartitionedCall
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityú
NoOpNoOp ^dense_4/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall^lstm_8/StatefulPartitionedCall^lstm_9/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ]: : : : : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2@
lstm_8/StatefulPartitionedCalllstm_8/StatefulPartitionedCall2@
lstm_9/StatefulPartitionedCalllstm_9/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs
ÃH
©

lstm_8_while_body_19542540*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3)
%lstm_8_while_lstm_8_strided_slice_1_0e
alstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0:	]N
;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0:	GI
:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0:	
lstm_8_while_identity
lstm_8_while_identity_1
lstm_8_while_identity_2
lstm_8_while_identity_3
lstm_8_while_identity_4
lstm_8_while_identity_5'
#lstm_8_while_lstm_8_strided_slice_1c
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensorJ
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource:	]L
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource:	GG
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:	¢/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp¢.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp¢0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpÑ
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ]   2@
>lstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shapeý
0lstm_8/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0lstm_8_while_placeholderGlstm_8/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]*
element_dtype022
0lstm_8/while/TensorArrayV2Read/TensorListGetItemÛ
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpReadVariableOp9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0*
_output_shapes
:	]*
dtype020
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOpð
lstm_8/while/lstm_cell_8/MatMulMatMul7lstm_8/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
lstm_8/while/lstm_cell_8/MatMulá
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0*
_output_shapes
:	G*
dtype022
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOpÙ
!lstm_8/while/lstm_cell_8/MatMul_1MatMullstm_8_while_placeholder_28lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!lstm_8/while/lstm_cell_8/MatMul_1Ð
lstm_8/while/lstm_cell_8/addAddV2)lstm_8/while/lstm_cell_8/MatMul:product:0+lstm_8/while/lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_8/while/lstm_cell_8/addÚ
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype021
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOpÝ
 lstm_8/while/lstm_cell_8/BiasAddBiasAdd lstm_8/while/lstm_cell_8/add:z:07lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 lstm_8/while/lstm_cell_8/BiasAdd
(lstm_8/while/lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2*
(lstm_8/while/lstm_cell_8/split/split_dim£
lstm_8/while/lstm_cell_8/splitSplit1lstm_8/while/lstm_cell_8/split/split_dim:output:0)lstm_8/while/lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2 
lstm_8/while/lstm_cell_8/splitª
 lstm_8/while/lstm_cell_8/SigmoidSigmoid'lstm_8/while/lstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2"
 lstm_8/while/lstm_cell_8/Sigmoid®
"lstm_8/while/lstm_cell_8/Sigmoid_1Sigmoid'lstm_8/while/lstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2$
"lstm_8/while/lstm_cell_8/Sigmoid_1¹
lstm_8/while/lstm_cell_8/mulMul&lstm_8/while/lstm_cell_8/Sigmoid_1:y:0lstm_8_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/while/lstm_cell_8/mul¡
lstm_8/while/lstm_cell_8/ReluRelu'lstm_8/while/lstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/while/lstm_cell_8/ReluÌ
lstm_8/while/lstm_cell_8/mul_1Mul$lstm_8/while/lstm_cell_8/Sigmoid:y:0+lstm_8/while/lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2 
lstm_8/while/lstm_cell_8/mul_1Á
lstm_8/while/lstm_cell_8/add_1AddV2 lstm_8/while/lstm_cell_8/mul:z:0"lstm_8/while/lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2 
lstm_8/while/lstm_cell_8/add_1®
"lstm_8/while/lstm_cell_8/Sigmoid_2Sigmoid'lstm_8/while/lstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2$
"lstm_8/while/lstm_cell_8/Sigmoid_2 
lstm_8/while/lstm_cell_8/Relu_1Relu"lstm_8/while/lstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2!
lstm_8/while/lstm_cell_8/Relu_1Ð
lstm_8/while/lstm_cell_8/mul_2Mul&lstm_8/while/lstm_cell_8/Sigmoid_2:y:0-lstm_8/while/lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2 
lstm_8/while/lstm_cell_8/mul_2
1lstm_8/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_8_while_placeholder_1lstm_8_while_placeholder"lstm_8/while/lstm_cell_8/mul_2:z:0*
_output_shapes
: *
element_dtype023
1lstm_8/while/TensorArrayV2Write/TensorListSetItemj
lstm_8/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add/y
lstm_8/while/addAddV2lstm_8_while_placeholderlstm_8/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/addn
lstm_8/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_8/while/add_1/y
lstm_8/while/add_1AddV2&lstm_8_while_lstm_8_while_loop_counterlstm_8/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm_8/while/add_1
lstm_8/while/IdentityIdentitylstm_8/while/add_1:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity¡
lstm_8/while/Identity_1Identity,lstm_8_while_lstm_8_while_maximum_iterations^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_1
lstm_8/while/Identity_2Identitylstm_8/while/add:z:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_2¶
lstm_8/while/Identity_3IdentityAlstm_8/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_8/while/NoOp*
T0*
_output_shapes
: 2
lstm_8/while/Identity_3¨
lstm_8/while/Identity_4Identity"lstm_8/while/lstm_cell_8/mul_2:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/while/Identity_4¨
lstm_8/while/Identity_5Identity"lstm_8/while/lstm_cell_8/add_1:z:0^lstm_8/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_8/while/Identity_5þ
lstm_8/while/NoOpNoOp0^lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/^lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp1^lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
lstm_8/while/NoOp"7
lstm_8_while_identitylstm_8/while/Identity:output:0";
lstm_8_while_identity_1 lstm_8/while/Identity_1:output:0";
lstm_8_while_identity_2 lstm_8/while/Identity_2:output:0";
lstm_8_while_identity_3 lstm_8/while/Identity_3:output:0";
lstm_8_while_identity_4 lstm_8/while/Identity_4:output:0";
lstm_8_while_identity_5 lstm_8/while/Identity_5:output:0"L
#lstm_8_while_lstm_8_strided_slice_1%lstm_8_while_lstm_8_strided_slice_1_0"v
8lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource:lstm_8_while_lstm_cell_8_biasadd_readvariableop_resource_0"x
9lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource;lstm_8_while_lstm_cell_8_matmul_1_readvariableop_resource_0"t
7lstm_8_while_lstm_cell_8_matmul_readvariableop_resource9lstm_8_while_lstm_cell_8_matmul_readvariableop_resource_0"Ä
_lstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensoralstm_8_while_tensorarrayv2read_tensorlistgetitem_lstm_8_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : 2b
/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp/lstm_8/while/lstm_cell_8/BiasAdd/ReadVariableOp2`
.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp.lstm_8/while/lstm_cell_8/MatMul/ReadVariableOp2d
0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp0lstm_8/while/lstm_cell_8/MatMul_1/ReadVariableOp: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
: 
Þ
¹
)__inference_lstm_9_layer_call_fn_19544157
inputs_0
unknown:	G
	unknown_0:
¢
	unknown_1:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_9_layer_call_and_return_conditional_losses_195409212
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
inputs/0
ü

I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_19544310

inputs
states_0
states_11
matmul_readvariableop_resource:	]3
 matmul_1_readvariableop_resource:	G.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02
MatMul_1/ReadVariableOp|
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
states/1
Ö
H
,__inference_dropout_9_layer_call_fn_19544201

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_dropout_9_layer_call_and_return_conditional_losses_195415182
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ¢:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 
_user_specified_nameinputs
¶

Ù
lstm_9_while_cond_19542360*
&lstm_9_while_lstm_9_while_loop_counter0
,lstm_9_while_lstm_9_while_maximum_iterations
lstm_9_while_placeholder
lstm_9_while_placeholder_1
lstm_9_while_placeholder_2
lstm_9_while_placeholder_3,
(lstm_9_while_less_lstm_9_strided_slice_1D
@lstm_9_while_lstm_9_while_cond_19542360___redundant_placeholder0D
@lstm_9_while_lstm_9_while_cond_19542360___redundant_placeholder1D
@lstm_9_while_lstm_9_while_cond_19542360___redundant_placeholder2D
@lstm_9_while_lstm_9_while_cond_19542360___redundant_placeholder3
lstm_9_while_identity

lstm_9/while/LessLesslstm_9_while_placeholder(lstm_9_while_less_lstm_9_strided_slice_1*
T0*
_output_shapes
: 2
lstm_9/while/Lessr
lstm_9/while/IdentityIdentitylstm_9/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_9/while/Identity"7
lstm_9_while_identitylstm_9/while/Identity:output:0*(
_construction_contextkEagerRuntime*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:
ß
Í
while_cond_19540221
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19540221___redundant_placeholder06
2while_while_cond_19540221___redundant_placeholder16
2while_while_cond_19540221___redundant_placeholder26
2while_while_cond_19540221___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
ø
à
$__inference__traced_restore_19544673
file_prefix2
assignvariableop_dense_4_kernel:	¢-
assignvariableop_1_dense_4_bias:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: ?
,assignvariableop_7_lstm_8_lstm_cell_8_kernel:	]I
6assignvariableop_8_lstm_8_lstm_cell_8_recurrent_kernel:	G9
*assignvariableop_9_lstm_8_lstm_cell_8_bias:	@
-assignvariableop_10_lstm_9_lstm_cell_9_kernel:	GK
7assignvariableop_11_lstm_9_lstm_cell_9_recurrent_kernel:
¢:
+assignvariableop_12_lstm_9_lstm_cell_9_bias:	#
assignvariableop_13_total: #
assignvariableop_14_count: %
assignvariableop_15_total_1: %
assignvariableop_16_count_1: <
)assignvariableop_17_adam_dense_4_kernel_m:	¢5
'assignvariableop_18_adam_dense_4_bias_m:G
4assignvariableop_19_adam_lstm_8_lstm_cell_8_kernel_m:	]Q
>assignvariableop_20_adam_lstm_8_lstm_cell_8_recurrent_kernel_m:	GA
2assignvariableop_21_adam_lstm_8_lstm_cell_8_bias_m:	G
4assignvariableop_22_adam_lstm_9_lstm_cell_9_kernel_m:	GR
>assignvariableop_23_adam_lstm_9_lstm_cell_9_recurrent_kernel_m:
¢A
2assignvariableop_24_adam_lstm_9_lstm_cell_9_bias_m:	<
)assignvariableop_25_adam_dense_4_kernel_v:	¢5
'assignvariableop_26_adam_dense_4_bias_v:G
4assignvariableop_27_adam_lstm_8_lstm_cell_8_kernel_v:	]Q
>assignvariableop_28_adam_lstm_8_lstm_cell_8_recurrent_kernel_v:	GA
2assignvariableop_29_adam_lstm_8_lstm_cell_8_bias_v:	G
4assignvariableop_30_adam_lstm_9_lstm_cell_9_kernel_v:	GR
>assignvariableop_31_adam_lstm_9_lstm_cell_9_recurrent_kernel_v:
¢A
2assignvariableop_32_adam_lstm_9_lstm_cell_9_bias_v:	
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
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
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

Identity_7±
AssignVariableOp_7AssignVariableOp,assignvariableop_7_lstm_8_lstm_cell_8_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8»
AssignVariableOp_8AssignVariableOp6assignvariableop_8_lstm_8_lstm_cell_8_recurrent_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¯
AssignVariableOp_9AssignVariableOp*assignvariableop_9_lstm_8_lstm_cell_8_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10µ
AssignVariableOp_10AssignVariableOp-assignvariableop_10_lstm_9_lstm_cell_9_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¿
AssignVariableOp_11AssignVariableOp7assignvariableop_11_lstm_9_lstm_cell_9_recurrent_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12³
AssignVariableOp_12AssignVariableOp+assignvariableop_12_lstm_9_lstm_cell_9_biasIdentity_12:output:0"/device:CPU:0*
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
AssignVariableOp_17AssignVariableOp)assignvariableop_17_adam_dense_4_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¯
AssignVariableOp_18AssignVariableOp'assignvariableop_18_adam_dense_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¼
AssignVariableOp_19AssignVariableOp4assignvariableop_19_adam_lstm_8_lstm_cell_8_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Æ
AssignVariableOp_20AssignVariableOp>assignvariableop_20_adam_lstm_8_lstm_cell_8_recurrent_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21º
AssignVariableOp_21AssignVariableOp2assignvariableop_21_adam_lstm_8_lstm_cell_8_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22¼
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_lstm_9_lstm_cell_9_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Æ
AssignVariableOp_23AssignVariableOp>assignvariableop_23_adam_lstm_9_lstm_cell_9_recurrent_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24º
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_lstm_9_lstm_cell_9_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_4_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¯
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_4_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¼
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_lstm_8_lstm_cell_8_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28Æ
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_lstm_8_lstm_cell_8_recurrent_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29º
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_lstm_8_lstm_cell_8_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¼
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_lstm_9_lstm_cell_9_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31Æ
AssignVariableOp_31AssignVariableOp>assignvariableop_31_adam_lstm_9_lstm_cell_9_recurrent_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32º
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_lstm_9_lstm_cell_9_bias_vIdentity_32:output:0"/device:CPU:0*
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
²

Ù
lstm_8_while_cond_19542212*
&lstm_8_while_lstm_8_while_loop_counter0
,lstm_8_while_lstm_8_while_maximum_iterations
lstm_8_while_placeholder
lstm_8_while_placeholder_1
lstm_8_while_placeholder_2
lstm_8_while_placeholder_3,
(lstm_8_while_less_lstm_8_strided_slice_1D
@lstm_8_while_lstm_8_while_cond_19542212___redundant_placeholder0D
@lstm_8_while_lstm_8_while_cond_19542212___redundant_placeholder1D
@lstm_8_while_lstm_8_while_cond_19542212___redundant_placeholder2D
@lstm_8_while_lstm_8_while_cond_19542212___redundant_placeholder3
lstm_8_while_identity

lstm_8/while/LessLesslstm_8_while_placeholder(lstm_8_while_less_lstm_8_strided_slice_1*
T0*
_output_shapes
: 2
lstm_8/while/Lessr
lstm_8/while/IdentityIdentitylstm_8/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm_8/while/Identity"7
lstm_8_while_identitylstm_8/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: ::::: 

_output_shapes
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
:ÿÿÿÿÿÿÿÿÿG:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG:

_output_shapes
: :

_output_shapes
:
¶[

D__inference_lstm_9_layer_call_and_return_conditional_losses_19543984

inputs=
*lstm_cell_9_matmul_readvariableop_resource:	G@
,lstm_cell_9_matmul_1_readvariableop_resource:
¢:
+lstm_cell_9_biasadd_readvariableop_resource:	
identity¢"lstm_cell_9/BiasAdd/ReadVariableOp¢!lstm_cell_9/MatMul/ReadVariableOp¢#lstm_cell_9/MatMul_1/ReadVariableOp¢whileD
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
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :¢2
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
B :¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2	
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
:ÿÿÿÿÿÿÿÿÿG2
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
valueB"ÿÿÿÿG   27
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
:ÿÿÿÿÿÿÿÿÿG*
shrink_axis_mask2
strided_slice_2²
!lstm_cell_9/MatMul/ReadVariableOpReadVariableOp*lstm_cell_9_matmul_readvariableop_resource*
_output_shapes
:	G*
dtype02#
!lstm_cell_9/MatMul/ReadVariableOpª
lstm_cell_9/MatMulMatMulstrided_slice_2:output:0)lstm_cell_9/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul¹
#lstm_cell_9/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_9_matmul_1_readvariableop_resource* 
_output_shapes
:
¢*
dtype02%
#lstm_cell_9/MatMul_1/ReadVariableOp¦
lstm_cell_9/MatMul_1MatMulzeros:output:0+lstm_cell_9/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/MatMul_1
lstm_cell_9/addAddV2lstm_cell_9/MatMul:product:0lstm_cell_9/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/add±
"lstm_cell_9/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_9_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_9/BiasAdd/ReadVariableOp©
lstm_cell_9/BiasAddBiasAddlstm_cell_9/add:z:0*lstm_cell_9/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_9/BiasAdd|
lstm_cell_9/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_9/split/split_dimó
lstm_cell_9/splitSplit$lstm_cell_9/split/split_dim:output:0lstm_cell_9/BiasAdd:output:0*
T0*d
_output_shapesR
P:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢*
	num_split2
lstm_cell_9/split
lstm_cell_9/SigmoidSigmoidlstm_cell_9/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid
lstm_cell_9/Sigmoid_1Sigmoidlstm_cell_9/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_1
lstm_cell_9/mulMullstm_cell_9/Sigmoid_1:y:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul{
lstm_cell_9/ReluRelulstm_cell_9/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu
lstm_cell_9/mul_1Mullstm_cell_9/Sigmoid:y:0lstm_cell_9/Relu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_1
lstm_cell_9/add_1AddV2lstm_cell_9/mul:z:0lstm_cell_9/mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/add_1
lstm_cell_9/Sigmoid_2Sigmoidlstm_cell_9/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Sigmoid_2z
lstm_cell_9/Relu_1Relulstm_cell_9/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/Relu_1
lstm_cell_9/mul_2Mullstm_cell_9/Sigmoid_2:y:0 lstm_cell_9/Relu_1:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢2
lstm_cell_9/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_9_matmul_readvariableop_resource,lstm_cell_9_matmul_1_readvariableop_resource+lstm_cell_9_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19543900*
condR
while_cond_19543899*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ¢   22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢*
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
:ÿÿÿÿÿÿÿÿÿ¢2
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
:ÿÿÿÿÿÿÿÿÿ¢2

IdentityÅ
NoOpNoOp#^lstm_cell_9/BiasAdd/ReadVariableOp"^lstm_cell_9/MatMul/ReadVariableOp$^lstm_cell_9/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿG: : : 2H
"lstm_cell_9/BiasAdd/ReadVariableOp"lstm_cell_9/BiasAdd/ReadVariableOp2F
!lstm_cell_9/MatMul/ReadVariableOp!lstm_cell_9/MatMul/ReadVariableOp2J
#lstm_cell_9/MatMul_1/ReadVariableOp#lstm_cell_9/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ã
Í
while_cond_19544050
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_16
2while_while_cond_19544050___redundant_placeholder06
2while_while_cond_19544050___redundant_placeholder16
2while_while_cond_19544050___redundant_placeholder26
2while_while_cond_19544050___redundant_placeholder3
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
B: : : : :ÿÿÿÿÿÿÿÿÿ¢:ÿÿÿÿÿÿÿÿÿ¢: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢:

_output_shapes
: :

_output_shapes
:

e
G__inference_dropout_8_layer_call_and_return_conditional_losses_19543509

inputs

identity_1^
IdentityIdentityinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identitym

Identity_1IdentityIdentity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿG:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ô

I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_19539998

inputs

states
states_11
matmul_readvariableop_resource:	]3
 matmul_1_readvariableop_resource:	G.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	]*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02
MatMul_1/ReadVariableOpz
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

MatMul_1l
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
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
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
split_
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2	
Sigmoidc
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
	Sigmoid_1\
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mulV
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
Reluh
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mul_1]
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
add_1c
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
	Sigmoid_2U
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
Relu_1l
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
mul_2d
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identityh

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

Identity_1h

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2

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
?:ÿÿÿÿÿÿÿÿÿ]:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_namestates
[

D__inference_lstm_8_layer_call_and_return_conditional_losses_19541970

inputs=
*lstm_cell_8_matmul_readvariableop_resource:	]?
,lstm_cell_8_matmul_1_readvariableop_resource:	G:
+lstm_cell_8_biasadd_readvariableop_resource:	
identity¢"lstm_cell_8/BiasAdd/ReadVariableOp¢!lstm_cell_8/MatMul/ReadVariableOp¢#lstm_cell_8/MatMul_1/ReadVariableOp¢whileD
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
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2
zeros`
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :G2
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
value	B :G2
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
:ÿÿÿÿÿÿÿÿÿG2	
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
strided_slice_2²
!lstm_cell_8/MatMul/ReadVariableOpReadVariableOp*lstm_cell_8_matmul_readvariableop_resource*
_output_shapes
:	]*
dtype02#
!lstm_cell_8/MatMul/ReadVariableOpª
lstm_cell_8/MatMulMatMulstrided_slice_2:output:0)lstm_cell_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul¸
#lstm_cell_8/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_8_matmul_1_readvariableop_resource*
_output_shapes
:	G*
dtype02%
#lstm_cell_8/MatMul_1/ReadVariableOp¦
lstm_cell_8/MatMul_1MatMulzeros:output:0+lstm_cell_8/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/MatMul_1
lstm_cell_8/addAddV2lstm_cell_8/MatMul:product:0lstm_cell_8/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/add±
"lstm_cell_8/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_8_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"lstm_cell_8/BiasAdd/ReadVariableOp©
lstm_cell_8/BiasAddBiasAddlstm_cell_8/add:z:0*lstm_cell_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
lstm_cell_8/BiasAdd|
lstm_cell_8/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell_8/split/split_dimï
lstm_cell_8/splitSplit$lstm_cell_8/split/split_dim:output:0lstm_cell_8/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG*
	num_split2
lstm_cell_8/split
lstm_cell_8/SigmoidSigmoidlstm_cell_8/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid
lstm_cell_8/Sigmoid_1Sigmoidlstm_cell_8/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_1
lstm_cell_8/mulMullstm_cell_8/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mulz
lstm_cell_8/ReluRelulstm_cell_8/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu
lstm_cell_8/mul_1Mullstm_cell_8/Sigmoid:y:0lstm_cell_8/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_1
lstm_cell_8/add_1AddV2lstm_cell_8/mul:z:0lstm_cell_8/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/add_1
lstm_cell_8/Sigmoid_2Sigmoidlstm_cell_8/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Sigmoid_2y
lstm_cell_8/Relu_1Relulstm_cell_8/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/Relu_1
lstm_cell_8/mul_2Mullstm_cell_8/Sigmoid_2:y:0 lstm_cell_8/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG2
lstm_cell_8/mul_2
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   2
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
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_8_matmul_readvariableop_resource,lstm_cell_8_matmul_1_readvariableop_resource+lstm_cell_8_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_19541886*
condR
while_cond_19541885*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿG:ÿÿÿÿÿÿÿÿÿG: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿG   22
0TensorArrayV2Stack/TensorListStack/element_shapeè
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG*
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
:ÿÿÿÿÿÿÿÿÿG2
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
:ÿÿÿÿÿÿÿÿÿG2

IdentityÅ
NoOpNoOp#^lstm_cell_8/BiasAdd/ReadVariableOp"^lstm_cell_8/MatMul/ReadVariableOp$^lstm_cell_8/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ]: : : 2H
"lstm_cell_8/BiasAdd/ReadVariableOp"lstm_cell_8/BiasAdd/ReadVariableOp2F
!lstm_cell_8/MatMul/ReadVariableOp!lstm_cell_8/MatMul/ReadVariableOp2J
#lstm_cell_8/MatMul_1/ReadVariableOp#lstm_cell_8/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¼
serving_default¨
I
lstm_8_input9
serving_default_lstm_8_input:0ÿÿÿÿÿÿÿÿÿ]?
dense_44
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:äº
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
1metrics
trainable_variables
regularization_losses
2layer_regularization_losses
3non_trainable_variables

4layers
		variables
5layer_metrics
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
;metrics
trainable_variables

<states
=layer_regularization_losses
>non_trainable_variables
regularization_losses

?layers
	variables
@layer_metrics
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
Ametrics
trainable_variables
Blayer_regularization_losses
Cnon_trainable_variables
regularization_losses

Dlayers
	variables
Elayer_metrics
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
Kmetrics
trainable_variables

Lstates
Mlayer_regularization_losses
Nnon_trainable_variables
regularization_losses

Olayers
	variables
Player_metrics
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
Qmetrics
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
regularization_losses

Tlayers
	variables
Ulayer_metrics
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	¢2dense_4/kernel
:2dense_4/bias
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
Vmetrics
"trainable_variables
Wlayer_regularization_losses
Xnon_trainable_variables
#regularization_losses

Ylayers
$	variables
Zlayer_metrics
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
,:*	]2lstm_8/lstm_cell_8/kernel
6:4	G2#lstm_8/lstm_cell_8/recurrent_kernel
&:$2lstm_8/lstm_cell_8/bias
,:*	G2lstm_9/lstm_cell_9/kernel
7:5
¢2#lstm_9/lstm_cell_9/recurrent_kernel
&:$2lstm_9/lstm_cell_9/bias
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
°
]metrics
7trainable_variables
^layer_regularization_losses
_non_trainable_variables
8regularization_losses

`layers
9	variables
alayer_metrics
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
°
bmetrics
Gtrainable_variables
clayer_regularization_losses
dnon_trainable_variables
Hregularization_losses

elayers
I	variables
flayer_metrics
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
&:$	¢2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
1:/	]2 Adam/lstm_8/lstm_cell_8/kernel/m
;:9	G2*Adam/lstm_8/lstm_cell_8/recurrent_kernel/m
+:)2Adam/lstm_8/lstm_cell_8/bias/m
1:/	G2 Adam/lstm_9/lstm_cell_9/kernel/m
<::
¢2*Adam/lstm_9/lstm_cell_9/recurrent_kernel/m
+:)2Adam/lstm_9/lstm_cell_9/bias/m
&:$	¢2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
1:/	]2 Adam/lstm_8/lstm_cell_8/kernel/v
;:9	G2*Adam/lstm_8/lstm_cell_8/recurrent_kernel/v
+:)2Adam/lstm_8/lstm_cell_8/bias/v
1:/	G2 Adam/lstm_9/lstm_cell_9/kernel/v
<::
¢2*Adam/lstm_9/lstm_cell_9/recurrent_kernel/v
+:)2Adam/lstm_9/lstm_cell_9/bias/v
ÓBÐ
#__inference__wrapped_model_19539923lstm_8_input"
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542473
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542814
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542092
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542117À
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
/__inference_sequential_4_layer_call_fn_19541577
/__inference_sequential_4_layer_call_fn_19542835
/__inference_sequential_4_layer_call_fn_19542856
/__inference_sequential_4_layer_call_fn_19542067À
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
ó2ð
D__inference_lstm_8_layer_call_and_return_conditional_losses_19543007
D__inference_lstm_8_layer_call_and_return_conditional_losses_19543158
D__inference_lstm_8_layer_call_and_return_conditional_losses_19543309
D__inference_lstm_8_layer_call_and_return_conditional_losses_19543460Õ
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
2
)__inference_lstm_8_layer_call_fn_19543471
)__inference_lstm_8_layer_call_fn_19543482
)__inference_lstm_8_layer_call_fn_19543493
)__inference_lstm_8_layer_call_fn_19543504Õ
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
Ì2É
G__inference_dropout_8_layer_call_and_return_conditional_losses_19543509
G__inference_dropout_8_layer_call_and_return_conditional_losses_19543521´
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
2
,__inference_dropout_8_layer_call_fn_19543526
,__inference_dropout_8_layer_call_fn_19543531´
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
ó2ð
D__inference_lstm_9_layer_call_and_return_conditional_losses_19543682
D__inference_lstm_9_layer_call_and_return_conditional_losses_19543833
D__inference_lstm_9_layer_call_and_return_conditional_losses_19543984
D__inference_lstm_9_layer_call_and_return_conditional_losses_19544135Õ
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
2
)__inference_lstm_9_layer_call_fn_19544146
)__inference_lstm_9_layer_call_fn_19544157
)__inference_lstm_9_layer_call_fn_19544168
)__inference_lstm_9_layer_call_fn_19544179Õ
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
Ì2É
G__inference_dropout_9_layer_call_and_return_conditional_losses_19544184
G__inference_dropout_9_layer_call_and_return_conditional_losses_19544196´
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
2
,__inference_dropout_9_layer_call_fn_19544201
,__inference_dropout_9_layer_call_fn_19544206´
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
E__inference_dense_4_layer_call_and_return_conditional_losses_19544237¢
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
*__inference_dense_4_layer_call_fn_19544246¢
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
ÒBÏ
&__inference_signature_wrapper_19542146lstm_8_input"
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
Ú2×
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_19544278
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_19544310¾
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
¤2¡
.__inference_lstm_cell_8_layer_call_fn_19544327
.__inference_lstm_cell_8_layer_call_fn_19544344¾
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
Ú2×
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_19544376
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_19544408¾
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
¤2¡
.__inference_lstm_cell_9_layer_call_fn_19544425
.__inference_lstm_cell_9_layer_call_fn_19544442¾
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
 £
#__inference__wrapped_model_19539923|+,-./0 !9¢6
/¢,
*'
lstm_8_inputÿÿÿÿÿÿÿÿÿ]
ª "5ª2
0
dense_4%"
dense_4ÿÿÿÿÿÿÿÿÿ®
E__inference_dense_4_layer_call_and_return_conditional_losses_19544237e !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¢
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_4_layer_call_fn_19544246X !4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ¢
ª "ÿÿÿÿÿÿÿÿÿ¯
G__inference_dropout_8_layer_call_and_return_conditional_losses_19543509d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿG
p 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿG
 ¯
G__inference_dropout_8_layer_call_and_return_conditional_losses_19543521d7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿG
p
ª ")¢&

0ÿÿÿÿÿÿÿÿÿG
 
,__inference_dropout_8_layer_call_fn_19543526W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿG
p 
ª "ÿÿÿÿÿÿÿÿÿG
,__inference_dropout_8_layer_call_fn_19543531W7¢4
-¢*
$!
inputsÿÿÿÿÿÿÿÿÿG
p
ª "ÿÿÿÿÿÿÿÿÿG±
G__inference_dropout_9_layer_call_and_return_conditional_losses_19544184f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¢
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¢
 ±
G__inference_dropout_9_layer_call_and_return_conditional_losses_19544196f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¢
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¢
 
,__inference_dropout_9_layer_call_fn_19544201Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¢
p 
ª "ÿÿÿÿÿÿÿÿÿ¢
,__inference_dropout_9_layer_call_fn_19544206Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ¢
p
ª "ÿÿÿÿÿÿÿÿÿ¢Ó
D__inference_lstm_8_layer_call_and_return_conditional_losses_19543007+,-O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
 Ó
D__inference_lstm_8_layer_call_and_return_conditional_losses_19543158+,-O¢L
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
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
 ¹
D__inference_lstm_8_layer_call_and_return_conditional_losses_19543309q+,-?¢<
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
0ÿÿÿÿÿÿÿÿÿG
 ¹
D__inference_lstm_8_layer_call_and_return_conditional_losses_19543460q+,-?¢<
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
0ÿÿÿÿÿÿÿÿÿG
 ª
)__inference_lstm_8_layer_call_fn_19543471}+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿGª
)__inference_lstm_8_layer_call_fn_19543482}+,-O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG
)__inference_lstm_8_layer_call_fn_19543493d+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿG
)__inference_lstm_8_layer_call_fn_19543504d+,-?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ]

 
p

 
ª "ÿÿÿÿÿÿÿÿÿGÔ
D__inference_lstm_9_layer_call_and_return_conditional_losses_19543682./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
 Ô
D__inference_lstm_9_layer_call_and_return_conditional_losses_19543833./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
 º
D__inference_lstm_9_layer_call_and_return_conditional_losses_19543984r./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿG

 
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¢
 º
D__inference_lstm_9_layer_call_and_return_conditional_losses_19544135r./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿG

 
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ¢
 «
)__inference_lstm_9_layer_call_fn_19544146~./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢«
)__inference_lstm_9_layer_call_fn_19544157~./0O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿG

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¢
)__inference_lstm_9_layer_call_fn_19544168e./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿG

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¢
)__inference_lstm_9_layer_call_fn_19544179e./0?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿG

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ¢Ë
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_19544278ý+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿG
"
states/1ÿÿÿÿÿÿÿÿÿG
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿG
EB

0/1/0ÿÿÿÿÿÿÿÿÿG

0/1/1ÿÿÿÿÿÿÿÿÿG
 Ë
I__inference_lstm_cell_8_layer_call_and_return_conditional_losses_19544310ý+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿG
"
states/1ÿÿÿÿÿÿÿÿÿG
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿG
EB

0/1/0ÿÿÿÿÿÿÿÿÿG

0/1/1ÿÿÿÿÿÿÿÿÿG
  
.__inference_lstm_cell_8_layer_call_fn_19544327í+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿG
"
states/1ÿÿÿÿÿÿÿÿÿG
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿG
A>

1/0ÿÿÿÿÿÿÿÿÿG

1/1ÿÿÿÿÿÿÿÿÿG 
.__inference_lstm_cell_8_layer_call_fn_19544344í+,-¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ]
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿG
"
states/1ÿÿÿÿÿÿÿÿÿG
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿG
A>

1/0ÿÿÿÿÿÿÿÿÿG

1/1ÿÿÿÿÿÿÿÿÿGÐ
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_19544376./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿG
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¢
# 
states/1ÿÿÿÿÿÿÿÿÿ¢
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ¢
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ¢
 
0/1/1ÿÿÿÿÿÿÿÿÿ¢
 Ð
I__inference_lstm_cell_9_layer_call_and_return_conditional_losses_19544408./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿG
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¢
# 
states/1ÿÿÿÿÿÿÿÿÿ¢
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿ¢
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿ¢
 
0/1/1ÿÿÿÿÿÿÿÿÿ¢
 ¥
.__inference_lstm_cell_9_layer_call_fn_19544425ò./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿG
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¢
# 
states/1ÿÿÿÿÿÿÿÿÿ¢
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ¢
C@

1/0ÿÿÿÿÿÿÿÿÿ¢

1/1ÿÿÿÿÿÿÿÿÿ¢¥
.__inference_lstm_cell_9_layer_call_fn_19544442ò./0¢
x¢u
 
inputsÿÿÿÿÿÿÿÿÿG
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿ¢
# 
states/1ÿÿÿÿÿÿÿÿÿ¢
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿ¢
C@

1/0ÿÿÿÿÿÿÿÿÿ¢

1/1ÿÿÿÿÿÿÿÿÿ¢Æ
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542092x+,-./0 !A¢>
7¢4
*'
lstm_8_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 Æ
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542117x+,-./0 !A¢>
7¢4
*'
lstm_8_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 À
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542473r+,-./0 !;¢8
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
J__inference_sequential_4_layer_call_and_return_conditional_losses_19542814r+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_sequential_4_layer_call_fn_19541577k+,-./0 !A¢>
7¢4
*'
lstm_8_inputÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_4_layer_call_fn_19542067k+,-./0 !A¢>
7¢4
*'
lstm_8_inputÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_4_layer_call_fn_19542835e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_sequential_4_layer_call_fn_19542856e+,-./0 !;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ]
p

 
ª "ÿÿÿÿÿÿÿÿÿ·
&__inference_signature_wrapper_19542146+,-./0 !I¢F
¢ 
?ª<
:
lstm_8_input*'
lstm_8_inputÿÿÿÿÿÿÿÿÿ]"5ª2
0
dense_4%"
dense_4ÿÿÿÿÿÿÿÿÿ