.PHONY: test fmt vet build bench

test:
	go test $(if $(PKG),$(PKG),./...)...

fmt:
	go fmt ./...

vet:
	go vet ./...

build:
	go build -o dipirona .

bench:
	go test ./pkg/model ./pkg/model/layer ./pkg/network -bench=. -benchtime=1s
