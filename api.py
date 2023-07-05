from utils import load_yaml
from pipline import Pipline

if __name__ == "__main__":
    cfg_path = 'configs/config.yaml'
    
    config = load_yaml(cfg_path)
    print(config["Tacotron2"]["config"])
    
    pipline = Pipline(
        config=config
    )
    text = " Theo kế hoạch, ngày 5/1 sẽ khai mạc kỳ họp bất thường lần thứ hai của Quốc hội khóa XV để xem xét một số vấn đề cấp bách, trong đó có quyết định về công tác nhân sự đại biểu và nhân sự khác. Nội dung về nhân sự sẽ được bố trí vào đầu kỳ họp."
    pipline.infer(text)