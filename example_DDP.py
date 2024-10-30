# DDP 이용해서 실험 돌리는 코드 프레임
# 이해 안되는 부분 있으면 언제든 질문 질문 
# 시작은 맨 밑에 if __name__ == "__main__": 부터 

import os, random, sys, socket, argparse, logging

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

# 로그 찍는놈 생성. wandb 쓰면 걔 써도 되는데 나는 logger를 선호함
# 얘 쓰면 로그를 따로 저장할 수 있음 + 어느 함수에서 이거 쓴다...머 이런거 알 수 있어서 좋음 
def logger_setup(log_file_name=None, log_file_folder_name = None, filepath=os.path.abspath(__file__), package_files=[]):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())

    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    info_file_handler = logging.FileHandler(log_file_folder_name + '/' + log_file_name , mode="a")
    info_file_handler.setLevel('INFO'.upper())
    info_file_handler.setFormatter(formatter)
    logger.addHandler(info_file_handler)

    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())
    return logger


# 파이썬 실행할 때 추가로 써주는 argument 들을 클래스 객체 형태(?)로 포장해주는 메소드
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    
    # 가령 머 실행할 때 --epochs 50 이런거 쓰면 args.epochs가 50이 됨
    parser.add_argument(
        "-e",
        "--epochs",
        default=50,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=2,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: %(default)s)"
    )
    # DDP 돌릴 때 사용할 포트 번호 (포트 번호에 관한 자세한 설명은 밑에 쭉 내리다보면 나옴)
    parser.add_argument(
        "--dist_port", type=int, default=6006, required=True, help="dist_port(default: %(default)s)"
    )
    
    parser.add_argument(
        "--seed", type=float, default=100, help="Set random seed for reproducibility"
    )
    
    args = parser.parse_args(argv)
    return args

# 폴더 생성해주는 메소드
def create_exp_folder(save_path) :
    try:
        os.mkdir(save_path)
    except:
        os.makedirs(save_path)

def train_one_epoch(model, train_dataloader, loss_function, optimizer, epoch, logger, node_rank = 0
):
    model.train()
    device = next(model.parameters()).device

    # data의 예제는 모...모가 있을까. 토큰? 텍스트? 
    for i, (data) in enumerate(train_dataloader):
        
        # 얘는 머냐면, backpropagation (역전파) 하다가 에러 터지는 경우가 많지 않나. 그럴 때 '야, 이 gradient 계산하다가 터졌어' 알려달라고 말해주게끔 세팅하는 코드임.
        with torch.autograd.detect_anomaly():
            data = data.to(device)

            optimizer.zero_grad()
            
            out_net = model(data)

            dist.barrier()

            loss = loss_function(out_net)
            loss.backward()

            optimizer.step()

        # node_rank == 0: 돌리는 프로세스가 n개라서 n번 로그 찍는걸 방지하고자 0번 gpu에서 돌아가는 프로세스만 로그 찍게 하는 조건문
        if i % 100 == 0 and node_rank == 0:
            logger.info(
                f"Train epoch {epoch + 1}: ["
                f"{i*len(img)}/{len(train_dataloader)*img.size(0)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {loss.item():.3f}'
            )

        dist.barrier()

# 제가 돌리는 모델은 인퍼런스 방식이 학습할 때 돌리는거랑 많이 달라서 (langauge model과 비슷한 느낌) 이건 머...알아서 짯시오 
def test_epoch(알아서_넣으시오):

    avg_performance = 0
    
    return avg_performance

# 4단계
def main(opts):
    
    node_rank = getattr(opts, "ddp.rank", 0)
    device_id = getattr(opts, "dev.device_id", torch.device("cpu"))

    logger = logger_setup(log_file_name = 'logs', log_file_folder_name = opts.save_path)
    
    ##############
    # 랜덤성 제어
    random.seed(opts.seed)
    np.random.seed(int(opts.seed))
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    ##############
    
    # 프로세스를 여러개 돌리기 때문에 로그찍는거 한 번 실행하면 n번 찍힘. 그거 방지하려고 돌리는 프로세스 중 0번 gpu를 쓰는 프로세스만 로그 찍게 만듦
    if node_rank == 0:
        logger.info("Create experiment save folder")

    train_dataset = 알아서_부르시오

    ######################################################
    # DDP는 아시다시피 GPU 개수에 맞춰 데이터셋을 N빵해서 각자 돌리는 기법. 그래서 학습 속도가 하나 쓸 때보다 빠른거고
    # 아래 코드들은 N빵해서 프로세스 별로 데이터 배치를 잘 배급하게끔 세팅하는 놈들이라 이해하면 편할듯
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)

    device = device_id

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler_train,
        num_workers=opts.num_workers,
        pin_memory=True,
    )
    ######################################################
    
    net = 알아서_부르시오    
    net = net.to(device)

    # DDP로 돌리고 있으니 그레디언트 잘 공유하면서 돌려라고 뉴럴넷에게 말해주는 하뭇
    net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[device_id],
            output_device=device_id,
            find_unused_parameters=True,
        )
    
    optimizer = 알아서_부르시오

    loss_function = 알아서_부르시오

    ########################################
    # test 준비
    test_dataset = 알아서_부르시오

    if node_rank == 0:  
        logger.info("Training mode : scratch!")
        logger.info(f"lambda : {opts.lmbda}")
        logger.info(f"milestones: {milestones}")

        logger.info(f"batch_size : {opts.batch_size}")
    ########################################

    best_performance = 0

    recent_saved_model_path = ''
    best_performance_model_path = ''
    
    # 처음부터 학습하면 last_epoch = 0, 이어서 학습하면 last_epoch = (특정 숫자)
    last_epoch = 0
    checkpoint = opts.checkpoint # 체크포인트 있으면 특정 경로, 없으면 None
    if checkpoint != "None":  # load from previous checkpoint

        if node_rank == 0:  
            logger.info(f"Loading {checkpoint}")

        checkpoint = torch.load(checkpoint)
        last_epoch = checkpoint["epoch"] + 1
        
        # 왜 이렇게까지 코드를 낭비하면서 부르는가? 
        # DDP 같이 멀티 gpu로 돌리는 애들은 멀티 지피유를 돌릴 수 있는 클래스의 멤버로 모델을 넣음
        # 그래서 state_dict를 뽑으면 module.~~ 식으로 paramter의 key가 뽑힘. 그래서 걍 불러오면 에러 터짐. module.~~ 거시기가 없는데요? 하면서
        # 그거 방지하려고 예외처리 막 하는 것
        try:
            try:
                net.load_state_dict(checkpoint["state_dict"])
            except:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_state_dict[k.replace("module.", "")] = v
                net.load_state_dict(new_state_dict)
        except:
            try:
                net.module.load_state_dict(checkpoint["state_dict"])
            except:
                new_state_dict = {}
                for k, v in checkpoint["state_dict"].items():
                    new_state_dict[k.replace("module.", "")] = v
                net.module.load_state_dict(new_state_dict)
            
        optimizer.load_state_dict(checkpoint["optimizer"])
        loss_function.load_state_dict(checkpoint["loss_function"])

        best_performance = checkpoint["best_performance"]

        recent_saved_model_path = checkpoint["recent_saved_model_path"]
        best_performance_model_path = checkpoint["best_performance_model_path"]
        
        del checkpoint

    save_path = opts.save_path
    
    # N개 프로세스는 각자 다른 프로그램이다. 뭔 뜻이냐면 얘들이 모든 순간에 같은 코드를 실행하고 있는건 아니라는 뜻이다
    # 예를 들어서 0번 프로세스가 100번 째 줄을 실행할 때 1번 프로세스는 80번 프로세스 실행하고...그러는 것
    # 그래서 '얘들아, 여기서 다같이 기다렸다가 모이면 이동하는거야' 라고 말해줄 필요가 있다. 그걸 해주는게 dist.barrier() 다. 
    # 각 프로세스에서 기록된 그레디언트를 공유해야 하기 때문에 모든 프로세스에 있는 뉴럴넷이 같은 수준의 학습을 진행하고 있어야 원하는대로 DDP가 실행되지 않겠는가 (내 뇌피셜이다)
    # 그리고 또, 얘는 어떤 프로세스가 오랜 시간동안 여기에 도달하지 못하면 '한 놈이 너무 안와요 뿌엥' 하면서 에러를 터뜨린다. 그건 특정 프로세스에서 뭔가 문제가 있다...는 뜻일 수 있다. 
    dist.barrier()

    for epoch in range(last_epoch, opts.epochs):
        
        train_one_epoch(net, train_dataloader, epoch, logger, node_rank)
        
        torch.cuda.empty_cache()

        dist.barrier()

        # 평가 하는 코드들
        # 평가 하는데 너무 오래 걸리면 '한 놈이 너무 안와요 뿌엥' 하면서 에러를 터뜨릴 수 있다. 그래서 평가는 다 같이 돌리되, 체크포인트 저장 등은 0번 프로세스만 하게끔 짰다.
        with torch.no_grad() :

            if (node_rank == 0):
                avg_performance = test_epoch(알아서_넣으시오)
            else:
                _, _, _, _, _ = test_epoch(알아서_넣으시오)
            
        torch.cuda.empty_cache()
        # 모델 저장
        if (node_rank == 0):
            try:
                state_dict = net.module.state_dict()
            except:
                state_dict = net.state_dict()

            if avg_performance < best_performance :
                best_performance = avg_performance

                try :
                    os.remove(best_performance_model_path) # 이전에 최고였던 모델 삭제 (지우는 이유: 용량 아끼려고 + 굳이 최고 성능이 아닌 모델을 저장할 이유가?)
                except :
                    logger.info("can not find prev_best_performance_model!")

                best_performance_model_path = save_path + '/' + f'best_perform_model_epoch_{epoch}_avg_{round(avg_performance, 5)}.pth'
                
                torch.save({
                    "epoch": epoch,
                    "state_dict": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "loss_function": loss_function.state_dict(),
                    "best_performance": best_performance,
                    "recent_saved_model_path":recent_saved_model_path,
                    "best_performance_model_path":best_performance_model_path
                    }, best_performance_model_path)

            # 이번 epoch에 학습한 친구도 저장하기
            try :
                os.remove(recent_saved_model_path) # 이전에 최고였던 모델 삭제
            except :
                logger.info("can not find recent_saved_model!")

            recent_saved_model_path = save_path + '/' + f'recent_model_epoch_{epoch}_avg_{round(avg_performance, 5)}.pth'

            torch.save({
                    "epoch": epoch,
                    "state_dict": state_dict,
                    "optimizer": optimizer.state_dict(),
                    "loss_function": loss_function.state_dict(),
                    "best_performance": best_performance,
                    "recent_saved_model_path":recent_saved_model_path,
                    "best_performance_model_path":best_performance_model_path
                    }, recent_saved_model_path)

        torch.cuda.empty_cache()
            
        dist.barrier()

# DDP 돌리기 위한 세팅 해주는 메소드
# DDP를 돌리기 위해 실행되는 n개 프로세스들은 특정 포트를 연결줄 삼아서 주고받는 TCP 통신을 이용해 모델의 그레디언트 정보 등을 공유한다고 생각하면 됨
# DDP 기법은 많긴 한데 여기서는 NCCL이란 기법을 씀. 이거 말고 또 있긴 한데 걔는 자꾸 에러 터지길래 화딱지 나서 쳐다도 안봄. 
# 근데 NCCL이 DDP 기법 중 뭔가 젤 구린(?) 느낌임. 아닐 수 있는데...암튼 머 그렇다. 
def distributed_init(opts) -> int:
    ddp_url = getattr(opts, "ddp.dist_url", None)

    node_rank = getattr(opts, "ddp.rank", 0)
    is_master_node = (node_rank == 0)

    if ddp_url is None:
        # 따로 지정한게 없어서 무조건 이쪽으로 들어옴
        ddp_port = opts.dist_port
        hostname = socket.gethostname()
        ddp_url = "tcp://{}:{}".format(hostname, ddp_port)
        setattr(opts, "ddp.dist_url", ddp_url)

    node_rank = getattr(opts, "ddp.rank", 0)
    world_size = getattr(opts, "ddp.world_size", 0)

    # 하나의 포트를 여러개의 DDP 프로세스 그룹이 같이 쓸 수 없음. 무조건 하나의 포트, 하나의 DDP 프로세스 그룹임.
    # 이거 몰라서 에러 터트린적 많으니 실험 여러개 돌릴 때 필히 포트 번호를 신경쓸 것
    if torch.distributed.is_initialized():
        print("DDP is already initialized and cannot be initialize twice!")
    else:
        print("distributed init (rank {}): {}".format(node_rank, ddp_url))

        dist_backend = getattr(opts, "ddp.backend", "nccl")  # "gloo"

        if dist_backend is None and dist.is_nccl_available():
            dist_backend = "nccl"
            if is_master_node:
                print(
                    "Using NCCL as distributed backend with version={}".format(
                        torch.cuda.nccl.version()
                    )
                )
        elif dist_backend is None:
            dist_backend = "gloo"

        dist.init_process_group(
            backend=dist_backend,
            init_method=ddp_url,
            world_size=world_size,
            rank=node_rank,
        )

        # perform a dummy all-reduce to initialize the NCCL communicator
        if torch.cuda.is_available():
            dist.all_reduce(torch.zeros(1).cuda())

    node_rank = torch.distributed.get_rank()
    setattr(opts, "ddp.rank", node_rank)
    return node_rank

# 3단계
# ddp용으로 돌리려는 프로세스들을 실행하겠다! 라고 이해하면 될 함수
def distributed_worker(i, main, opts):
    setattr(opts, "dev.device_id", i)
    torch.cuda.set_device(i)
    setattr(opts, "dev.device", torch.device(f"cuda:{i}"))

    # 내가 돌리려는 gpu 중 번호가 가장 낮은 애는 0, 그 다음 애는 1...이렇게 지정됨 (예. 4~7번 GPU를 동시에 돌린다면 4번 gpu의 ddp rank가 0이 되는거임)
    ddp_rank =  i
    setattr(opts, "ddp.rank", ddp_rank)
    
    # DDP 돌리기 위한 세팅을 여기서 함. 보다 자세한 설명은 distributed_init()에 달린 주석을...
    node_rank = distributed_init(opts)
    setattr(opts, "ddp.rank", node_rank)
    main(opts)

# 2단계
# 인식되는 gpu 개수 인식하고 그 개수에 맞게 n개 프로세스를 실행시킴
def ddp_or_single_process(argvs):

    opts = parse_args(argvs)

    # 학습하다가 중간에 꺼지는 경우가 있을 수 있음. 그럴 때 이어서 학습시키기 위한 의도로 다음과 같이 짬
    checkpoint = "None"

    if opts.seed is not None:
        save_path = f'./checkpoint/exp_epochs_{opts.epochs}_seed_{opts.seed}_batch_size_{opts.batch_size}_learning_rate_{opts.lr}'
        
        if os.path.exists(save_path):
            logger = logger_setup(log_file_name = 'logs', log_file_folder_name = save_path)
            logger.info("find checkpoint...")
            
            file_list = os.listdir(save_path)

            for file_name in file_list:
                # 가장 최근 에폭을 학습한 모델명을 recent_model 어쩌구로 지정했고, 저장하는 체크포인트 확장자를 .pth로 함 (확장자는 .pt로 해도 되고 머 아무거나 써도 되긴 함)
                # 여기 조건문에 걸리면 이제 '아, 학습하고있던 모델이 있구나' 라고 프로그램이 인지함
                if ('recent_model' in file_name) and ('.pth' in file_name): 
                    logger.info(f"checkpoint exist, name: {file_name}")
                    checkpoint = f'{save_path}/{file_name}'
            
            if checkpoint == 'None':
                logger.info("no checkpoint is here")
            
        else:
            create_exp_folder(save_path)
            logger = logger_setup(log_file_name = 'logs', log_file_folder_name = save_path)
            logger.info("Create new exp folder!")

        logger.info(f"seed : {opts.seed}")
        logger.info(f"exp name : exp_epochs_{opts.epochs}_seed_{opts.seed}_batch_size_{opts.batch_size}_learning_rate_{opts.lr}")

    # 내가 입력했던 argument 갖고 포장한 상자(?)에 몇 가지 더 구겨넣는다고 생각하면 됨.
    setattr(opts, "checkpoint", checkpoint) # 학습하던 모델이 있다면 걔 경로가 여기에 들어감
    setattr(opts, "save_path", save_path) # 학습하는 모델을 저장하는 경로가 여기에 들어감

    setattr(opts, "dev.num_gpus", torch.cuda.device_count()) # 내가 쓰려고 하는 gpu 개수
    setattr(opts, "ddp.world_size", torch.cuda.device_count()) # 솔직히 먼지 잘 모르는데 남들이 gpu 개수로 지정했길래 나도 똑같이 함

    logger.info(f"opts: {opts}")
    
    # 얘가 이제 DDP 돌리려고 프로세스 n개 부르는 함수
    torch.multiprocessing.spawn(
            fn=distributed_worker,
            args=(main, opts),
            nprocs=getattr(opts, "dev.num_gpus"),
        )

if __name__ == "__main__":
    # 1단계
    # py 실행할 때 같이 넣은 argument 들이 'sys.argv[1:]' 에 들어있다고 생각하면 됨. 
    ddp_or_single_process(sys.argv[1:])
