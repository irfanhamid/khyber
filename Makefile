SUBDIRS = examples/cpuid src/lib src/unittest

.PHONY: subdirs $(SUBDIRS)

subdirs: $(SUBDIRS)

$(SUBDIRS):
	make -C $@

all: $(SUBDIRS)
