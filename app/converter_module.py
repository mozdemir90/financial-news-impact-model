import json
import csv

def json_to_csv(input_path, output_path):
    """
    Belirtilen JSON dosyasını (içinde bir nesne listesi barındıran) okur
    ve CSV formatına dönüştürür.
    
    Args:
        input_path (str): Girdi JSON dosyasının yolu.
        output_path (str): Çıktı CSV dosyasının oluşturulacağı yol.
        
    Returns:
        bool: İşlem başarılıysa True, değilse False döner.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f_json:
            veri_listesi = json.load(f_json)
        
        # Eğer JSON dosyası boşsa veya içinde liste yoksa işlem yapma
        if not veri_listesi or not isinstance(veri_listesi, list):
            print(f"Uyarı: '{input_path}' boş veya beklenen formatta değil.")
            return False

        # CSV başlıklarını ilk elemanın anahtarlarından al
        basliklar = veri_listesi[0].keys()

        with open(output_path, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=basliklar)
            writer.writeheader()
            writer.writerows(veri_listesi)
            
        return True

    except FileNotFoundError:
        print(f"Hata: '{input_path}' dosyası bulunamadı.")
        return False
    except Exception as e:
        print(f"CSV dönüştürme sırasında bir hata oluştu: {e}")
        return False