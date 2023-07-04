from utils import load_yaml
from pipline import Pipline

if __name__ == "__main__":
    # text = "aa1 iz1 a1 iz1 <spc> k u5 ngz <spc> k o3 <spc> m oo6 tc <spc> h i2 nhz <spc> d u1 ngz <spc> ch u1 ngz <spc> v ee2 <spc> k a3 iz3 <spc> v i6 nhz <spc> h a6 <spc> l o1 ngz <spc> dd a5 <spc> th aa1 nc <spc> k w e1 nc <spc> . dd ee3 nc <spc> m uw3 kc <spc> k o3 <spc> ph aa2 nc <spc> k u5 <spc> k i5 <spc> ."

    cfg_path = 'configs/config.yaml'
    
    config = load_yaml(cfg_path)
    print(config["Tacotron2"]["config"])
    
    pipline = Pipline(
        config=config
    )
    text = " Theo kế hoạch, ngày 5/1 sẽ khai mạc kỳ họp bất thường lần thứ hai của Quốc hội khóa XV để xem xét một số vấn đề cấp bách, trong đó có quyết định về công tác nhân sự đại biểu và nhân sự khác. Nội dung về nhân sự sẽ được bố trí vào đầu kỳ họp."
    pipline.infer(text)