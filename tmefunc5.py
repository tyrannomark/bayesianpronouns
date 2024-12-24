import functools
import collections
import operator

Test = []
class F:
  def __init__(self,f):
    self.F                      = f
  def ensure(g):
    if not isinstance(g,F) and isinstance(g, collections.abc.Callable): return( F(g) )
    return( g )
  def __call__(self,*args):
    r                           = self.F(*args)
    if isinstance(r, collections.abc.Callable): return F.ensure( r )
    return( r )
  def compose(fromA,toB):
    def f(*args): return( F.ensure(toB)(F.ensure(fromA)(*args)) )
    return( F(f) )
  def __or__(self,other): return F.compose(self,F.ensure(other))
  ## def __ror__(self,other): return F.compose(F.ensure(other),self)
  def __and__(self,other): return F.compose(F.ensure(other),self)
  ## def __rand__(self,other): return F.compose(self,F.ensure(other))
  def __mul__(self,other):
    def f(*args): return( (self(*args),other(*args)) )
    return( F( f ) )
  def V(v):
    def Vf(*args): return( v )
    return( F(Vf) )
  def reargument(self,f):
    def newf(*args): return( self.F(*f(*args)) )
    return( F(newf) )
  def curry2nd(self,second):
    return self.reargument( lambda *args: ( args[0], second ) + args[1:] ) 
  def unTube(self):
    def f(*args): return( [ args ] )
    return( self.reargument(f) )
  def toTube(self):
    def f(args): return( args )
    return( self.reargument(f) )
  def partial(self,*args): return( F(functools.partial(self.F,*args)) )
  def fold(self,zero=None):
    def f(l):
      n = len(l)
      if n == 0: return( zero )
      if n == 1: return( l[0] )
      n2 = int( (n+1) / 2 )
      left = self.fold(zero)(l[:n2]); right = self.fold(zero)(l[n2:])
      return( self.F( left, right ) )
    return( F(f) )
  def foldl(self,zero=None):
    def f(l): return( self.foldl(self.F(zero,l[0]))(l[1:]) if l else zero )
    return( F(f) )
  def foldr(self,zero=None):
    def f(l): return( self(l[0],self.foldr(zero)(l[1:])) if l else zero )
    return( F(f) )
  def default(self,zero=None): return(
      F(lambda param: F(lambda state: F(lambda l: self.F(param,l) if state == zero else state)))
  )
  def map(self): return( F(lambda l: list( map(self.F,l) ) ) )
  def mapi(self): return( F(lambda l: list( map(lambda i: self.F(l[i],i),range(len(l))) ) ) )
  def smap(self,zero=None):
    @F
    def ff(l):
      nonlocal zero
      zero = self.F(zero,l)
      return zero
    return ff.map()
  def mapKeys(self): return( F(lambda d: { self.F(k):d[k] for k in d } ) )
  def mapValues(self): return( F(lambda d: { k:self.F(d[k]) for k in d } ) )
  def grep(self,pattern=None):
    if pattern == None:  return( F(lambda l: [ i for i in l if self(i) ]) )
    return( F(lambda l: [ i for i in l if self(i,pattern) ]) )
  def onIf(self,cnd):
    return( F( lambda arg: self(arg) if cnd(arg) else arg ) )
def compose(fromA,toB):
  def f(*args): return( toB(fromA(*args)) )
  return( F(f) )

Test.append( compose(lambda x: x*2, lambda x: x+3) )
def composeL(fromA,toB):
  def f(*args): return( F.ensure(toB).toTube().F(fromA(*args)) )
  return( F(f) )

Test.append( lambda x: (composeL(lambda x,y: (x*y,x+y), lambda x,y: x-y)(2,3)) )
Id = F(lambda x: x)
Cl = F(compose).foldr(Id)
C  = Cl.unTube()
D  = F(compose).reargument(lambda x,y: [y,x]).foldr(Id).unTube()
CL = F(composeL).foldr(Id).unTube()
DL = F(composeL).reargument(lambda x,y: [y,x]).foldr(Id).unTube()
def compositionTest():
  def mkDouble(m):
    def double(a): print(m); return( (a+a) )
    return( double )
  print( C(mkDouble(1),mkDouble(2),mkDouble(3))(5) )
  print( D(mkDouble(1),mkDouble(2),mkDouble(3))(5) )
  def mkMulplus(m):
    def mulplus(a,b): print(m); return( (a*b,a+b) )
    return( mulplus )
  ## print( CL(mkMulplus(1),mkMulplus(2),mkMulplus(3))(2,3) )
  ## print( DL(mkMulplus(1),mkMulplus(2),mkMulplus(3))(2,3) )
## compositionTest()
## print( list( map( lambda i: Test[i](1), range(len(Test)) ) ) )
ljoin                           = F(operator.concat).foldr([])
zipfmin = F(lambda f, v1,v2: F(lambda i: f(v1[i],v2[i])).map()(
  range(min(len(v1),len(v2)))
))
zipfmax = F(lambda f, v1,v2: F(lambda i: f(v1[i] if i < len(v1) else None,
                                         v2[i] if i < len(v2) else None)).map()(
  range(max(len(v1),len(v2)))
))
zip                             = zipfmin.partial(lambda x1,x2: (x1,x2))
zip3 = F( lambda w: [ (w[0][i],w[1][i],w[2][i]) for i in range(len(w[0])) ] )
get = F( lambda n: F( lambda i: i[n] ) )
unzip3 = F( lambda ts: ( get(0).map()( ts ), get(1).map()( ts ), get(2).map()( ts ) ) )

@F
def grep(f): return( F(lambda l: F.ensure(f).grep(l)) )
@F
def splitOnCondition(f,keep=False):
  def splitOnConditionF(l):
    pieces = [ [] ]
    def breakl(ll):
      nonlocal pieces, keep
      # print("breakl",ll)
      return( pieces.append([ll],[]) if keep else pieces.append( [] ) )
    def extendl(ll):
      nonlocal pieces
      # print("extendl",ll)
      return( pieces[len(pieces)-1].append(ll) )
    lmap(lambda ll: breakl(ll) if f(ll) else extendl(ll), l)
    return( pieces )
  return( F(splitOnConditionF) )

@F
def dict2values(d): return( list(d.values()) )
@F
def dict2keys(d): return( list(d.keys()) )

@F
def dict2keyvalue(d): return( zip(d.keys(),d.values()) )

@F
def list2dict(keyFn,valueFn):
  @F
  def list2dictF(ll):
    d = {}
    for i in range(len(ll)):
      l = ll[i]
      k,v = keyFn(l),valueFn(l)
      if k not in d: d[k] = [v]
      else: d[k].append(v)
    return( d )
  return( list2dictF )
@F
def list2dicti(keyFn,valueFn):
  @F
  def list2dictF(ll):
    d = {}
    for i in range(len(ll)):
      l = ll[i]
      k,v = keyFn(l,i),valueFn(l,i)
      if k not in d: d[k] = [v]
      else: d[k].append(v)
    return( d )
  return( list2dictF )
pairs2listdict = list2dict(lambda v: v[0],
                           lambda v: v[1])
map2listdict = F( lambda f: list2dict(f,lambda x,i: x) )

@F
def map2listlist(f):
  return( map2listdict(f) | F(lambda v: v.values()) )

@F
def mergeDicts(dicts):
  merged = {}
  for d in dicts:
    for k in d.keys():
      if k in merged: continue
      merged[k] = d[k]
  return( merged )
@F
def dictUnify2(a,b):
  u = {}
  u.update(a)
  u.update(b)
  return( u )
dictUnify = dictUnify2.foldr({})
@F
def collate(fn):
  collection = []
  encoder = {}
  @F
  def update(a):
    nonlocal collection, encoder, fn
    b = fn(a)
    if b in encoder: return encoder[b]
    l = len(collection)+1
    collection.append( (l,b) )
    encoder[b] = l
    return( l )
  @F
  def reln():
    nonlocal collection
    return( collection )
  @F
  def encode(a):
    nonlocal encoder
    return( encoder[ fn(a) ] )
  return( (update,reln,encode) )
@F
def testCollate():
  fn = F( lambda s: s )
  (cupdate,creln,cencode) = collate( fn )
  text = ["them","pieces","are","no","good","them"]
  cupdate.map()(text)
  print( cencode.map()(text) )
@F
def readXmlFile(filepath):
  return( xml.etree.ElementTree.parse( filepath ).getroot() )
enpath = F( lambda z: [ (0, z) ] )

@F
def listPaths(path):
  children = path[-1][1].findall("./*")
  subpaths = list(map( lambda newleaf: path+[newleaf],zip(range(1,1+len(children)),children) ))
  result = [ path ]
  for subpath in subpaths: result += listPaths( subpath )
  return( result )
@F
def pathsByFinalTagRe(pattern):
  def f(paths): return( [ path for path in paths if re.match(pattern, path[-1][1].tag) ] )
  return( F(f) )
## (readXmlFile | enpath | listPaths)(filepath)
@F
def Tuple(*fns):
  @F
  def t(*args): return( tuple(map(lambda f: f(*args), fns)) )
  return t
@F
def Const(value): return( F(lambda x: value) )
@F
def Pr(format="%s"):
  @F
  def f(x):
    print( format % (str(x)) )
    return( x )
  return( f )
@F
def Prep(format="%s"):
  @F
  def f(x):
    print( format % (repr(x)) )
    return( x )
  return( f )
@F
def Prl(format="%d"):
  @F
  def f(x):
    print( format % (len(x)) )
    return( x )
  return( f )

@F
def Prf( f=Id ):
    @F
    def ff( x ):
        print( f(x) )
        return x
    return ff

flatten = F( lambda ll: [ i for l in ll for i in l ] )

concatStrings = F(lambda l: F(lambda a, b: a+b).fold("")( l ))

get = F( lambda k: F( lambda l: l[k] ) )
last = F( lambda n: F( lambda ll: ll[-n:] ) )
appendS = F( lambda s: F( lambda i: i+s ).map() )
span = F( lambda fromI, toI: F( lambda ll: ll[ fromI: toI ] ))
dictValue = F( lambda k: F( lambda d: d[ k ] ) )

classMethod0 = F( lambda method: F( lambda obj: obj.__class__.__dict__[ method ](obj) ) )
classMethod1 = F( lambda method, a1: F( lambda obj: obj.__class__.__dict__[ method ](obj,a1) ) )
classMethod2 = F( lambda method, a1, a2: F( lambda obj: obj.__class__.__dict__[ method ](obj,a1,a2) ) )

@F
def doPass( f ):
    @F
    def ff( x ):
        f( x )
        return x
    return ff
