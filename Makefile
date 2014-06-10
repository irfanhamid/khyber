SUBDIRS = examples/cpuid src/lib src/unittest benchmarks/sqrt

.PHONY: subdirs $(SUBDIRS)

subdirs: $(SUBDIRS)

$(SUBDIRS):
	make -C $@

all: $(SUBDIRS)
