SUBDIRS = examples/cpuid

.PHONY: subdirs $(SUBDIRS)

subdirs: $(SUBDIRS)

$(SUBDIRS):
	make -C $@

all: $(SUBDIRS)
