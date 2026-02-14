# Binance-Testnet-Trade-Bot

# Trade Bot Proje Özeti

Bu projede hedefim “AI mucizesi” yaratmak değildi. Daha gerçekçi hedefler koydum:

Binance üzerinde çalışan bir trade botu yazmak (Python).

Stratejiyi basit tutup, risk yönetimini güçlü yapmak.

Üretimde (real money) koşmadan önce Testnet üzerinde denemek.

Botu gözlemleyebilmek için okunur, tek satırlık log çıktısı üretmek.

Ücret (fee) ve slippage (kayma) gibi “gerçek hayat” maliyetlerini hesaba katmak.

Sonuç: Kârdan çok, “böyle bir sistem nasıl kurulur, nasıl test edilir, nasıl güvenli hale getirilir?” sorularına cevap veren güzel bir mini proje çıktı.

# Bot Nasıl Çalışıyor?
1) Basit bir strateji: Bollinger Bands (BB)

İlk botumun temelinde, piyasada çok bilinen bir gösterge var: Bollinger Bands.

Basitleştirilmiş mantık:

Fiyat BB lower altına sarkarsa: “aşırı satım olabilir” → AL

Fiyat tekrar BB mid üzerine dönerse: “ortalama dönüş” → SAT

Bu, mucize değil; ama test etmeye ve otomasyon kurmaya uygun, anlaşılır bir başlangıç.

2) Risk yönetimi: Stop-loss + Günlük maksimum zarar kill-switch

Bence botun en değerli kısmı strateji değil, fren sistemi:

Stop-loss (zarar-kes): Pozisyondayken belirli bir seviyede otomatik çıkış.

Günlük maksimum zarar (DD) kill-switch: Gün içinde zarar belli bir eşiği aşarsa bot o gün yeni işlem açmayı durduruyor.

Bu sayede “yanlış giden bir günde” botun inatla işlem açıp batması yerine, kendini kapatıp ertesi güne bırakması hedeflendi.

3) Fee ve slippage hesabı

Kâğıt üstünde “al-sat” basit görünür ama gerçek hayatta:

Her işlemde komisyon ödersiniz.

Market emirlerinde çoğu zaman taker fee devreye girer.

Emirler defterde dolarken slippage oluşur.

Bot bu maliyetleri yaklaşık şekilde hesaba katıyor. Özellikle küçük bütçelerde bu maliyetler stratejinin kârlılığını ciddi etkiliyor.
