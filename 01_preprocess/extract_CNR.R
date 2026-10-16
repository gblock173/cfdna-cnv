#============================================================#
# 기본 정보 설정
COMMENT = "
Filename        : extract_CNR.R
Modification    :
Description     : Copy Number Ratio를 추출하는 스크립트

주요 기능:
- QDNAseq를 이용한 CNR 추출
- CNR 시각화
- 결과 저장
"

#============================================================#
# 패키지 불러오기
#============================================================#
suppressPackageStartupMessages(
  {
    library(QDNAseq)
    library(QDNAseq.hg38)
    library(optparse)
    library(Biobase)
  }
)
library(GenomeInfoDb)
#============================================================#
# 명령행 인자 파싱
#============================================================#
option_list <- list(
  make_option(c("--bam"), type = "character", default = NULL, help = "Path to bam file. Required."),
  make_option(c("--id"), type = "character", default = NULL, help = "Analysis ID. Default: [%default]"),
  make_option(c("--binSize"), type = "numeric", default = 100, help = "Bin Size. Default is 100Kb bin. Default: [%default]"),
  make_option(c("--outDir"), type = "character", default = "./", help = "Output Directory. Default: [%default]")
)

parseobj = OptionParser(option_list = option_list)
opt = parse_args(parseobj)

if (is.null(opt$bam)) {
  print_help(parseobj)
  stop(domain = NA)
}

options(scipen = 0,
        stringsAsFactors = F)

bam_path = opt$bam
sample_id = opt$id
binSize = opt$binSize
outDir = opt$outDir
dir.create(outDir, recursive = TRUE, showWarnings = FALSE)
unit = "kb"

set.seed(1234)

#============================================================#
# 사용 예시
#============================================================#
# Rscript 01_preprocess/extract_CNR.R --bam <BAM 파일 경로> --id <샘플ID> --binSize <bin 크기(Kb)> --outDir <출력 디렉토리>
# 예시:
example = "
Rscript 01_preprocess/extract_CNR.R \
  --bam sample.bam \
  --id example \
  --binSize 100 \
  --outDir ./results/
"

#============================================================#
# QDNAseq 수행
#============================================================#
bins = getBinAnnotations(binSize = binSize, genome = 'hg38')

readCounts = binReadCounts(bins,
                           bamfiles = bam_path,
                           pairedEnds = TRUE)

readCountsFiltered = applyFilters(readCounts)

readCountsFiltered = estimateCorrection(readCountsFiltered)

readCountsFiltered = applyFilters(readCountsFiltered,
                                  chromosomes = "Y")
#correct
copyNumbers = correctBins(readCountsFiltered)

#normalize
copyNumbersNormalized = normalizeBins(copyNumbers)

#smooth
copyNumbersSmooth = smoothOutlierBins(copyNumbersNormalized)

figure_out = file.path(outDir, paste0(sample_id, ".", binSize, "kb.genome-wide.cnv.png"))

png(figure_out)
plot(copyNumbersSmooth)
dev.off()

#============================================================#
# CNV 세그먼트 및 call 추가
#============================================================#
# segment
copyNumbersSegmented = segmentBins(copyNumbersSmooth)

# call
copyNumbersCalled = callBins(copyNumbersSegmented)

# 세그먼트 plot
figure_seg = file.path(outDir, paste0(sample_id, ".", binSize, unit, ".genome-wide.cnv.segmented.png"))
png(figure_seg)
plot(copyNumbersCalled)
dev.off()

# call 값만 추출하여 저장
bin_info <- fData(copyNumbersCalled)
calls_values <- assayDataElement(copyNumbersCalled, "calls")
call_df <- data.frame(chromosome = bin_info$chromosome,
                      start = bin_info$start,
                      end = bin_info$end,
                      calls = calls_values[, 1])
output_file <- file.path(outDir, paste0(sample_id, ".", binSize, unit, ".cnv.calls.txt"))

# 숫자 지수 표기 방지 및 정수형 변환
options(scipen=999)
call_df$start <- as.integer(call_df$start)
call_df$end <- as.integer(call_df$end)

# feature 컬럼 추가 및 컬럼 순서 변경
call_df$feature <- paste0(call_df$chromosome, ":", call_df$start, "-", call_df$end)
call_df <- call_df[, c("feature", "chromosome", "start", "end", "calls")]

write.table(call_df, # 전체 데이터프레임을 저장
            file = output_file,
            sep = "\t", row.names = FALSE, quote = FALSE)
#============================================================#
# 결과 저장 (기존)
#============================================================#

exportBins(object = copyNumbersSmooth,
           file = file.path(outDir, paste0(sample_id, ".", binSize, unit, ".cnv.log2.txt")),
           format = "tsv",
           logTransform = T)

exportBins(object = copyNumbersSmooth,
           file = file.path(outDir, paste0(sample_id, ".", binSize, unit, ".cnv.raw.txt")),
           format = "tsv",
           logTransform = F)

exportBins(object = copyNumbersSmooth,
           file = file.path(outDir, paste0(sample_id, ".", binSize, unit, ".cnv.log2.igv")),
           format = "igv",
           logTransform = T)
