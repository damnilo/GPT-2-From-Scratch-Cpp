# Plan razvoja

Redosled je obavezan. Širi model dolazi tek kad testovi, validacija i konfiguracija stoje. Postojeći `Training Files/model.bin` je verzija 1 ovog formata i ostaje učitljiv za generisanje. Novi podaci u čekpointu dižu verziju fajla.

Trenutno stanje na koje se plan naslanja:

* `Model` uvek pravi tri `Transformer` bloka (`Transformer/src/Model.cpp`).
* `CrossEntropyLoss::loss` je zbir po prozoru, ne srednja vrednost po tokenu.
* `AdamW` drži momente `m` i `v` i korak `timestep`, a `Checkpoint` snima samo konfiguraciju i težine.
* `Tensor::generator` i generator u `generate` koriste `std::random_device`.
* Putanja do podataka je `Training Files` pored radnog direktorijuma, ili apsolutna Windows putanja u `main.cpp`.
* Ako `model.bin` postoji, program preskače trening i odmah generiše.
* CUDA se uključuje samo za `matmul` i `batchMatmul` kad je proizvod dimenzija bar 65536 (`NumCPP/src/Tensor.cpp`).

## 01. Test suite i gradient checking

Prioritet: veoma visok.

Nema testova. CMake već postoji, pa je CTest dovoljan i ne uvodi novu biblioteku. GoogleTest ostaje rezerva ako kasnije zatrebaju bogatiji izveštaji. NumPy služi samo da proizvede referentne brojeve. Sami C++ testovi čitaju te brojeve iz fajlova u repou, tako da čista instalacija bez Pythona i dalje prolazi `ctest`.

Uradi:

* `tests/` sa jednim test izvršnim fajlom registrovanim preko `add_test` u `CMakeLists.txt`.
* `tests/reference/` sa NumPy skriptom koja za fiksiran ulaz upisuje očekivane tenzore. Skripta se pokreće ručno kad se promeni operacija, ne u svakom `ctest` pozivu.
* Slučajeve za sabiranje, oduzimanje, množenje i deljenje, uključujući broadcasting gde se oblik jedne ose razlikuje i gde je jedna osa dužine 1.
* `matmul` i `batchMatmul` na malim poznatim matricama, plus jedan oblik blizu izlaznog sloja, reda `[64, 32] @ [32, V]`.
* Forward i backward za `Embedding`, `Linear`, `LayerNorm`, `GeLU`, `Softmax`, `CrossEntropyLoss`, `Attention`, `MultiHeadAttention`, `FeedForward`, `Transformer` i `PositionalEmbedding`.
* Numeričku proveru gradijenta: pomeri jedan parametar za `1e-3`, uporedi konačnu razliku sa analitičkim gradijentom. Za složene blokove proveri nekoliko težina, ne svaku.
* Poseban slučaj za `PositionalEmbedding::backward`: gradijent težina mora biti upisan, a povratna vrednost prazna, bez sabiranja dva prazna tenzora.

Tolerancije, upisane u testu pored svake grupe:

| Operacija | Apsolutna tolerancija |
| --- | --- |
| Elementarne operacije na CPU | `1e-5` |
| `matmul` / `batchMatmul` na CPU | `1e-4` |
| Isti proizvod preko cuBLAS-a, ako je test pokrenut sa CUDA-om | `1e-3` |
| Analitički gradijent naspram konačne razlike | `1e-2` relativno, ili `1e-3` apsolutno na malim težinama |
| Softmax i cross-entropy | `1e-5` na CPU |

cuBLAS ne mora da se poklopi bit po bit sa OpenMP petljom. Test koji to zahteva bi bio pogrešan.

Kriterijum završetka: `ctest` prolazi na čistoj instalaciji posle `cmake -S . -B build && cmake --build build && ctest --test-dir build`, a tolerancije iz tabele stoje u ovom fajlu i u kodu testa.

## 02. Pouzdano treniranje i evaluacija

Prioritet: veoma visok.

Broj koji se danas štampa nije uporediv između prozora različite dužine, niti između dva pokretanja. Poslednji prozor epohe je kraći od 64, a seed ne postoji.

Uradi:

* Podeli kodirani korpus jednom, pre treninga. Zadnjih 5% tokena je validacija, ostatak je trening. Granica se računa iz dužine i upisuje u log, da dva pokretanja istog fajla dobiju isti rez.
* `CrossEntropyLoss` i dalje može da vrati zbir, ali u petlji deli taj zbir brojem predviđenih tokena u prozoru. U log ide ta srednja vrednost, u natima po tokenu. Kad loss postane srednja vrednost, stopa učenja `0.001` više nije ista kao na zbiru od ~64 tokena. U istom koraku odluku o stopi zapiši u log, da kriva ne bude tiho promenjena.
* Jedan `uint64` seed prosleđen u `Tensor::generator` i u generator za uzorkovanje. Podrazumevano `1`. Inicijalizacija težina i redosled uzoraka tada mogu da se ponove.
* Verzija čekpointa `2`. Pored težina snimi `timestep`, momente `m` i `v`, broj završene epohe, broj koraka i seed. Učitavanje verzije `1` i dalje diže samo model, bez nastavka optimizatora.
* Posle svake epohe ispiši srednji trening loss i srednji validacioni loss na istom obliku prozora. Validacija ne zove `step` i ne menja težine.
* Tri fiksna prompta, iste token nizove u svakoj epohi, i uzorak od 64 tokena sa temperaturama zapisanim u logu. Tekst služi da se vidi da li validacioni loss i čitljivost idu zajedno.

Kriterijum završetka: jedan log pokazuje krivu trening i validacionog loss-a po epohama. Ako validacioni loss krene da raste dok trening i dalje pada, to je preučenost i tako se i kaže. Ako oba i dalje padaju, model još uči i preučenosti nema.

## 03. Konfigurabilan model i trening

Prioritet: visok.

Drugi korisnik danas mora da otvori `main.cpp` da promeni širinu, broj epoha ili putanju.

Uradi:

* Proširi `ModelConfig` poljem `n_layers`. `Model` pravi toliko blokova umesto tri zakucana poziva. `n_heads` i dalje mora da deli `embedding_dim`.
* Jedan konfiguracioni fajl, tekstualan, sa `n_layers`, `embedding_dim`, `n_heads`, `block_size`, `batch_size`, `learning_rate`, `weight_decay`, `epochs`, `seed` i putanjama do korpusa, spajanja i čekpointa. Podrazumevane vrednosti su današnje: 3, 32, 4, 64, 1, `0.001`, `0.01`.
* Putanje su uvek relativne u odnosu na trenutni direktorijum ili na putanju datu u konfiguraciji. Apsolutna `C:/Users/HP/...` putanja izlazi iz `main.cpp`.
* Tri pokretanja, bez menjanja izvora:

```text
GPT_2_From_Scratch train --config train.txt
GPT_2_From_Scratch resume --config train.txt
GPT_2_From_Scratch generate --config train.txt --tokens 200
```

`train` kreće iz početka i odbija da pregazi postojeći čekpoint osim uz eksplicitnu zastavicu. `resume` učitava verziju 2, uključujući AdamW. `generate` samo učitava težine i fiksne promptove.

Kriterijum završetka: na drugoj mašini, posle kloniranja, `cmake` builda i jedne `train` komande sa priloženim malim `train.txt`, krene kratak eksperiment. U izvoru se ništa ne menja.

## 04. Benchmark CPU naspram CUDA

Prioritet: visok.

Merenje mora da pogodi pravi put. Sa pragom 65536 mali proizvodi ostanu na OpenMP-u i tabela bi lažno rekla da CUDA ništa ne radi.

Uradi:

* Poseban program `bench`, ne deo treninga. Za svaki slučaj ispiši koji je put izvršen: jedna nit, OpenMP, ili cuBLAS.
* Veličine: mala pažnja `[1, 64, 8] @ [1, 8, 64]`, feed-forward `[64, 32] @ [32, 128]`, izlazni sloj `[64, 32] @ [32, 5256]`, i isti oblici na širini 128.
* Četiri merenja: `matmul`, `batchMatmul`, jedan forward i backward kroz `Model`, i jedan pun korak (forward, loss, backward, `AdamW`).
* Svako merenje se zagreje, pa se uzme medijana od najmanje 10 ponavljanja. U red tabele idu dimenzije, CPU, GPU ako postoji, broj niti, medijana u milisekundama i ubrzanje u odnosu na jednu nit.
* Ako `nvcc` nije nađen, CUDA redovi se ne popunjavaju. Ne pretpostavljaj ubrzanje.

Kriterijum završetka: tabela u `ROADMAP.md` ili u `bench/RESULTS.md` sa dimenzijama, hardverom, vremenima i ubrzanjem. CUDA put se zove bržim samo u redovima gde je medijana zaista manja.

## 05. Veći model i bolja optimizacija

Prioritet: srednji. Počinje tek kad 01–04 prođu.

Današnji model širine 32 je završen eksperiment. Sledeći služi da se vidi da li kapacitet i recept treninga pomeraju validacioni loss.

Uradi, svaku promenu posebno, na istom validacionom rezu iz faze 02:

* `embedding_dim` 64, zatim 128. Broj glava i dalje deli širinu. Kontekst prvo ostaje 64, pa se proba 128.
* Weight tying: ista težina za tabelu tokena i izlazni `Linear`, sa jednim gradijentom koji sabira oba doprinosa.
* Raspored stope učenja: linearno zagrevanje, pa kosinus do nule. Uporedi sa konstantnom stopom na istom seedu.
* Dropout `0.1` na izlazu pažnje i feed-forward mreže, pre sabiranja sa residual vezom, samo ako validacioni loss u širem modelu krene da raste dok trening loss pada. Tokom `generate` dropout je isključen.
* Preklapajući prozori, pomeraj 32 umesto 64, kao poseban eksperiment. Ne mešaj ga u istom pokretanju sa promenom širine.

Svaki pokušaj ostavlja jedan red: šta je promenjeno, trening loss, validacioni loss, i uzorak na tri fiksna prompta.

Kriterijum završetka: tabela tih redova pokazuje kako kapacitet i konfiguracija treninga menjaju validacioni loss i tekst. Ako širina 128 ne spusti validaciju u odnosu na 32, to je rezultat i tu se staje dok se recept ne popravi.

## Šta se ne radi usput

* Ne širiti model da bi se „popravio“ tekst pre faze 02. Bez validacionog loss-a nema čime da se ta promena meri.
* Ne pomerati kaznu u `generate` kao zamenu za fazu 05. Uzorkovanje na širini 32 je već na granici tog modela.
* Ne brisati učitavanje čekpointa verzije 1. Stari `model.bin` i dalje služi za generisanje.
