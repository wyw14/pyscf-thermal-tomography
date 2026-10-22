"""Independent AO stationarity, explicit ERIs and exhaustive box-QP validation.

No candidate or author reference imports. Shared PySCF molecular integrals are
checked by explicit four-index contractions. Small path optima enumerate all
physical correspondences, with a different QP method from the author solver.
"""
from copy import deepcopy
import itertools
import numpy as np
from scipy import linalg, optimize
from pyscf import gto, scf, lib

lib.num_threads(1)


def basis(n):
    result=[]
    for i in range(n):
        a=np.zeros((n,n),complex);a[i,i]=1;result.append(a)
    for i in range(n):
        for j in range(i+1,n):
            a=np.zeros((n,n),complex);a[i,j]=a[j,i]=2**-.5;result.append(a)
            a=np.zeros((n,n),complex);a[i,j]=1j*2**-.5;a[j,i]=-1j*2**-.5;result.append(a)
    return np.asarray(result)


def coordinates(matrices, frame):
    return np.einsum('kij,...ji->...k',frame,matrices).real


def fermion_entropy(p):
    occupations=np.linalg.eigvalsh(p)
    if occupations.min()<=0 or occupations.max()>=1:
        raise ValueError('Pauli violation')
    return float(-np.sum(occupations*np.log(occupations)+(1-occupations)*np.log1p(-occupations)))


def log_odds(p):
    value=linalg.logm(p)-linalg.logm(np.eye(len(p))-p)
    return (value+value.conj().T)/2


def explicit_ghf(mol,dm):
    n=mol.nao_nr();eri=mol.intor('int2e')
    charge=dm[:n,:n]+dm[n:,n:]
    coulomb=np.einsum('ijkl,lk->ij',eri,charge)
    result=np.zeros_like(dm,dtype=complex)
    for spin1 in range(2):
        for spin2 in range(2):
            block=dm[spin1*n:(spin1+1)*n,spin2*n:(spin2+1)*n]
            result[spin1*n:(spin1+1)*n,spin2*n:(spin2+1)*n]=-np.einsum('iklj,kl->ij',eri,block)
            if spin1==spin2:
                result[spin1*n:(spin1+1)*n,spin2*n:(spin2+1)*n]+=coulomb
    return result


def record_from_density(case,p,order,directions):
    dm=case['inverse_root']@p@case['inverse_root']
    values=np.einsum('aij,ji->a',case['generators'],dm).real
    u=case['unlabeled']; values[u]=values[u[order]]
    rows=[]
    for dp in directions:
        ddm=case['inverse_root']@dp@case['inverse_root']
        moments=np.einsum('aij,ji->a',case['generators'],ddm).real
        moments[u]=moments[u[order]]
        rows.append(np.r_[moments,-np.trace(log_odds(p)@dp).real])
    return {'values':values,'entropy':fermion_entropy(p),'directions':np.asarray(rows).reshape((len(directions),len(values)+1))}


def build_case(kind='path',count=4,queries=2):
    spec={'atom':'He 0 0 0; H 0 0 1.2','charge':1,'basis':'sto-3g','verbose':0}
    if kind=='two':spec={'atom':'H 0 0 0','spin':1,'basis':'sto-3g','verbose':0}
    if kind=='six':spec={'atom':'H 0 0 0; H 0 0 1.1; H 0.3 0.6 2.0','charge':1,'basis':'sto-3g','verbose':0}
    if kind=='eight':spec={'atom':'H 0 0 0; H 0 0 1.2; H 0.5 0.3 2.2; H 0.8 0.2 3.4','charge':0,'basis':'sto-3g','verbose':0}
    mol=gto.M(**spec);mf=scf.GHF(mol);s=mf.get_ovlp();n=len(s)
    root=np.asarray(linalg.fractional_matrix_power(s,.5),complex);invroot=np.linalg.inv(root)
    canonical=basis(n);keep=list(range(n-1))+list(range(n,n*n))
    generators=np.einsum('ij,ajk,kl->ail',root,canonical[keep],root)
    constraints=np.asarray([s]);targets=np.array([mol.nelectron],float)
    unlabeled=np.array([3,4,13,14],int) if kind in ('path','active','long') else np.array([],int)
    interval=(.4,1.6) if n==4 else (.8,2.)
    coupling=.85 if n==4 else .25
    rng=np.random.default_rng(104729)
    points=np.array([[.08,-.045,.035,.025,-.105,-.085],[.025,-.115,.035,.09,-.04,-.085],[.105,-.035,.035,.04,-.08,-.085],[.04,-.085,.035,.11,-.025,-.085]])
    sigma=np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
    ps=[];dps=[]
    for index in range(count):
        if n==4:
            p=np.eye(n,dtype=complex)/2
            point=points[0] if kind in ('active','long') else points[index%4]
            p[:2,:2]+=np.einsum('a,aij->ij',point[:3],sigma)
            p[2:,2:]+=np.einsum('a,aij->ij',point[3:],sigma)
            if kind in ('complex','gauge','constraint'):
                u,_=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
                p=u@p@u.conj().T
        else:
            u,_=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))
            p=(u*(mol.nelectron/n+np.linspace(-.05,.05,n)))@u.conj().T
        directions=[]
        for query in range(queries):
            dp=np.zeros((n,n),complex)
            if query!=0 or index==count-1:
                if kind in ('path','active','long'):
                    dp[:2,:2]=np.einsum('a,aij->ij',rng.normal(size=3)*.004,sigma)
                    dp[2:,2:]=np.einsum('a,aij->ij',rng.normal(size=3)*.004,sigma)
                else:
                    z=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));dp=.002*(z+z.conj().T);dp-=np.eye(n)*np.trace(dp)/n
            directions.append(dp)
        ps.append(p);dps.append(np.asarray(directions).reshape((queries,n,n)))
    if kind=='constraint':
        # A second nontrivial fixed moment is enforced at every frame.
        op=canonical[0]-canonical[1]
        target=float(np.trace(op@ps[0]).real)
        for index in range(count):
            delta=(np.trace(op@ps[index]).real-target)/2
            ps[index]=ps[index]-delta*op
            for query in range(queries):
                dps[index][query]-=.5*np.trace(op@dps[index][query]).real*op
        constraints=np.concatenate((constraints,(root@op@root)[None]));targets=np.r_[targets,target]
    if kind=='gauge':
        generators=np.concatenate((generators,(.7*generators[0]-.4*generators[1])[None],s[None]))
    case={'spec':spec,'mol':mol,'mf':mf,'s':s,'root':root,'inverse_root':invroot,'generators':generators,'constraints':constraints,'targets':targets,'unlabeled':unlabeled,'interval':interval,'coupling':coupling,'ps':ps,'dps':dps,'kind':kind}
    orders=[rng.permutation(len(unlabeled)) for _ in ps];case['orders']=orders
    case['records']=[record_from_density(case,p,order,dp) for p,order,dp in zip(ps,orders,dps)]
    first=physical_modes(case,case['records'][0])[0]
    temperature=2.8 if kind=='active' else 1.1
    case['anchor']=first['c']+temperature*first['n']
    if kind not in ('active','long'):case['anchor']+=np.r_[rng.normal(size=len(generators))*.01,0.]
    ad=rng.normal(size=(queries,len(generators)+1))*.01
    if queries:ad[0]=0
    case['anchor_directions']=ad
    # Every generated query must actually lie in the declared fixed-constraint tangent space.
    for directions in dps:
        for dp in directions:
            residual=np.einsum('aij,ji->a',constraints,invroot@dp@invroot).real
            if np.max(np.abs(residual))>1e-11:
                raise AssertionError(('Invalid author query: fixed moments change',residual))
    return case


def physical_modes(case,record):
    mf=case['mf'];n=len(case['s']);frame=basis(n)
    cmat=coordinates(case['constraints'],frame).T
    vmat=coordinates(case['generators'],frame).T
    remove=np.eye(n*n)-cmat@linalg.pinv(cmat)
    wsystem=remove@vmat
    coeff_null=linalg.null_space(wsystem,rcond=1e-10)
    gauge=coeff_null@coeff_null.T
    anchor=np.asarray(case.get('anchor',np.zeros(len(case['generators'])+1)))
    eq=np.vstack((cmat.T,vmat.T))
    candidates=[]
    for mapping in itertools.permutations(range(len(case['unlabeled']))):
        values=np.asarray(record['values']).copy();u=case['unlabeled'];values[u]=values[u[np.asarray(mapping,int)]]
        rhs=np.r_[case['targets'],values]
        dc,_,_,_=linalg.lstsq(eq,rhs,lapack_driver='gelsy')
        if np.max(np.abs(eq@dc-rhs))>2e-9:continue
        dm=np.einsum('a,aij->ij',dc,frame);p=case['root']@dm@case['root']
        try:
            entropy=fermion_entropy(p)
            if abs(entropy-record['entropy'])>2e-9:continue
        except ValueError:continue
        fock=mf.get_hcore()+case['coupling']*explicit_ghf(case['mol'],dm)
        thermal=case['root']@log_odds(p)@case['root']
        w0=linalg.lstsq(wsystem,-remove@coordinates(fock,frame),lapack_driver='gelsy',cond=1e-10)[0]+gauge@anchor[:-1]
        wt=linalg.lstsq(wsystem,-remove@coordinates(thermal,frame),lapack_driver='gelsy',cond=1e-10)[0]
        joint=linalg.null_space(np.column_stack((wsystem,remove@coordinates(thermal,frame))),rcond=1e-10)
        candidates.append({'mapping':mapping,'density':dm,'p':p,'c':np.r_[w0,0.],'n':np.r_[wt,1.],'projector':joint@joint.T,'fock':fock,'thermal':thermal})
    if not candidates:raise ValueError('Independent physical feasibility failed')
    return candidates


def box_system(path,anchor):
    n=len(path);d=len(anchor);a=np.zeros((n*d,n));b=np.empty(n*d)
    for j,mode in enumerate(path):
        sl=slice(j*d,(j+1)*d);a[sl,j]=mode['n'];b[sl]=mode['c']-(anchor if j==0 else path[j-1]['c'])
        if j:a[sl,j-1]=-path[j-1]['n']
    return a,b


def exhaustive(case):
    layers=[physical_modes(case,row) for row in case['records']];solutions=[]
    for path in itertools.product(*layers):
        a,b=box_system(path,case['anchor'])
        fit=optimize.lsq_linear(a,-b,bounds=case['interval'],method='trf',lsq_solver='exact',tol=1e-13,max_iter=300)
        if not fit.success:raise RuntimeError('Independent TRF box-QP failed')
        z=np.array([mode['c']+t*mode['n'] for mode,t in zip(path,fit.x)])
        objective=float(np.sum((a@fit.x+b)**2))
        solutions.append({'objective':objective,'parameters':z,'path':path,'temperature_derivative':2*a.T@(a@fit.x+b),'mapping_key':tuple(v['mapping'] for v in path)})
    solutions.sort(key=lambda x:(x['objective'],x['mapping_key']))
    return {'best':solutions[0],'gap':solutions[1]['objective']-solutions[0]['objective'] if len(solutions)>1 else None,'candidate_counts':[len(x) for x in layers],'enumerated':len(solutions)}


def perturb(case,query,step):
    result=dict(case)
    result['anchor']=case['anchor']+step*case['anchor_directions'][query]
    result['records']=[record_from_density(case,p+step*dp[query],order,np.zeros((0,len(p),len(p)))) for p,dp,order in zip(case['ps'],case['dps'],case['orders'])]
    return result


def finite_response(case,query,step=2e-5):
    lower=exhaustive(perturb(case,query,-step))['best'];upper=exhaustive(perturb(case,query,step))['best']
    if lower['mapping_key']!=upper['mapping_key']:raise AssertionError('Independent derivative left its discrete branch')
    return {'parameters':(upper['parameters']-lower['parameters'])/(2*step),'objective':(upper['objective']-lower['objective'])/(2*step),'density':(np.array([x['density'] for x in upper['path']])-np.array([x['density'] for x in lower['path']]))/(2*step)}


def arguments(case,records=None,checkpoint=None):
    kwargs={'unlabeled_indices':case['unlabeled'].tolist(),'temperature_interval':case['interval'],'coupling':case['coupling']}
    if checkpoint is None:kwargs.update(initial_anchor=case['anchor'],initial_directions=case['anchor_directions'])
    else:kwargs['checkpoint']=checkpoint
    return [case['spec'],case['generators'],case['constraints'],case['targets'],case['records'] if records is None else records],kwargs
